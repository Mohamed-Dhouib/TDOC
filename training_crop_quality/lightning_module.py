import math
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from model_cropped import DocMDetectorConfig, DocMDetector
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.functional as F
import os
import gc

def focal_loss_from_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "sum"
    ) -> torch.Tensor:
    """Multi-class focal loss for imbalanced classification."""
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()

    loss = -alpha * (1 - pt).pow(gamma) * log_pt

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        return loss

class DocMDetectorPLModule(pl.LightningModule):
    """Lightning module for G_theta bounding box quality training (Section 3.2)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.accumulate_grad_batches=self.config.accumulate_grad_batches

        self.model = DocMDetector(
            config=DocMDetectorConfig(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,

            )
        )
        
    def training_step(self, batch, batch_idx):
        opt   = self.optimizers()
        sched = self.lr_schedulers()

        if batch_idx==0 or not hasattr(self, 'loss'):
            self.loss=[]

        grad_accumulation=self.accumulate_grad_batches
        backprop_now = ((batch_idx+1)% grad_accumulation) == 0

        PIXEL_BUDGET = int(1.5*64*256 * 128 * 512)
        total_samples = sum(el["image"].shape[0] for el in batch)
        sum_loss     = 0.0

        for index_batch,el in enumerate(batch):

            images,top_bottom,left_right,image_mask,top_bottom_mask,left_right_mask,gts = el["image"],el["top_bottom"],el["left_right"],el["image_mask"],el["top_bottom_mask"],el["left_right_mask"],el["gt"]  
            G, C, H, W = images.shape
            max_chunk = max(1, PIXEL_BUDGET // (H*W))

            for start in range(0, G, max_chunk):
                end = min(G, start + max_chunk)
                micro_imgs   = images[start:end]
                micro_top_bottom = top_bottom[start:end] 
                micro_left_right=left_right[start:end] 

                micro_image_mask   = image_mask[start:end]         
                micro_tb_mask      = top_bottom_mask[start:end]    
                micro_lr_mask      = left_right_mask[start:end]    
                
                
                micro_labels = gts[start:end].to(micro_imgs.device)
                g = end - start


                logits = self.model.encoder(micro_imgs,micro_top_bottom,micro_left_right,micro_image_mask, micro_tb_mask, micro_lr_mask)

                if self.config.focal_loss:
                    chunk_loss = focal_loss_from_logits(
        logits, micro_labels, gamma=2.0, alpha=0.25, reduction="sum"
    )
                else :
                    chunk_loss = F.cross_entropy(logits, micro_labels, reduction="sum")
                
                sum_loss += chunk_loss.item()

                loss_to_backprop = chunk_loss / (total_samples*grad_accumulation)

                is_last = (start + max_chunk) >= G and len(batch)==index_batch+1
                
                ddp_model = self.trainer.strategy.model

                if is_last and backprop_now:
                    self.manual_backward(loss_to_backprop)

                else :
                    with ddp_model.no_sync():
                        self.manual_backward(loss_to_backprop)

                

        if backprop_now:
            opt.step()
            sched.step()
            opt.zero_grad()
            if ((batch_idx+1) % (grad_accumulation*100)) == 0 :
                    gc.collect()
                    torch.cuda.empty_cache()

        avg_loss = sum_loss / total_samples
        self.loss.append(avg_loss)
        if len(self.loss)>1000:
            self.loss=self.loss[1:]

        self.log_dict({"loss_avg": sum(self.loss)/len(self.loss)}, sync_dist=True , prog_bar=True)
        
        self.log("lr", sched.get_last_lr()[0], on_step=True, prog_bar=True,sync_dist=True,logger=False)

            

    def configure_optimizers(self):

        max_iter = None

        try :
            nodes= int(os.getenv('SLURM_JOB_NUM_NODES'))
            if nodes<1 :
                nodes=1
        except :
            nodes=1
        print(f"optimize detected {nodes} nodes")

        assert self.config.get("max_epochs", -1)>0

        if getattr(self.config, "scheduler_num_training_samples_per_cycle_and_epoch", -1) == -1:
            samples_per_cycle = len(self.trainer.datamodule.train_dataset)
        else:
            samples_per_cycle = self.config.scheduler_num_training_samples_per_cycle_and_epoch

        max_iter = (self.config.max_epochs * samples_per_cycle) / (
            self.config.train_batch_size * torch.cuda.device_count() * nodes * self.config.accumulate_grad_batches
        )
        max_iter=int(max_iter)
        print(f"max iter is set to {max_iter}")
        assert max_iter is not None
        
        for _,param in self.model.named_parameters():
            param.requires_grad=True

        if (nodes*torch.cuda.device_count())>1 :
            optimizer = ZeroRedundancyOptimizer(
                    [param for _,param in self.model.named_parameters() if   param.requires_grad==True ],
                    optimizer_class=torch.optim.AdamW,
                    lr=self.config.lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(
                [p for _, p in self.model.named_parameters() if p.requires_grad],
                lr=self.config.lr,
                weight_decay=0.01
            )
                
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        """Cosine annealing scheduler with linear warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, [lr_lambda])

def make_custom_collate_fn(bin_size=1):
    """Create collate function that groups samples by size bins."""

    def custom_collate_fn(batch):
        """
        1) Flatten nested lists of [tensor, gt] pairs.
        2) Assign each sample to a (h_bin, w_bin) where h_bin = ceil(H/bin_size)*bin_size, w_bin = ceil(W/bin_size)*bin_size.
        3) Within each bin, pad all images to that bin’s actual max H and W.
        4) Return a list of sub‐batches, each a dict with keys "image" and "gt".
        """
        flat = [item for sublist in batch for item in sublist]

        bins = {}
        for tensor, top_bottom, left_right, gt in flat:
            _, H, W = tensor.shape
            h_bin = math.ceil(H / bin_size) * bin_size
            w_bin = math.ceil(W / bin_size) * bin_size

            bins.setdefault((h_bin, w_bin), []).append((tensor, top_bottom, left_right, gt))

        sub_batches = []
        for (h_bin, w_bin), items in bins.items():
            max_h = max(tensor.shape[1] for tensor, _, _, _ in items)
            max_w = max(tensor.shape[2] for tensor, _, _, _ in items)

            max_h = max(16, max_h)
            max_w = max(16, max_w) 


            max_w_tp = max(top_bottom.shape[2] for _, top_bottom, _, _ in items)
            max_h_lr = max(left_right.shape[1] for _, _, left_right, _ in items)

            def pad_to_target(img: torch.Tensor):
                c, h, w = img.shape
                pad_h = max_h - h
                pad_w = max_w - w
                return F.pad(img, (0, pad_w, 0, pad_h))

            def pad_to_target_w(img):
                c, h, w = img.shape
                pad_w = max_w_tp - w
                return F.pad(img, (0, pad_w, 0, 0))

            def pad_to_target_h(img):
                c, h, w = img.shape
                pad_h = max_h_lr - h
                return F.pad(img, (0, 0, 0, pad_h))

            images = torch.stack([pad_to_target(tensor) for tensor, _, _, _ in items], dim=0)
            top_bottom = torch.stack([pad_to_target_w(top_bottom) for _, top_bottom, _, _ in items], dim=0)
            left_right = torch.stack([pad_to_target_h(left_right) for _, _, left_right, _ in items], dim=0)

            image_masks = torch.stack([  
                F.pad(
                    torch.ones((tensor.shape[1], tensor.shape[2]), dtype=torch.bool),  
                    (0, max_w - tensor.shape[2], 0, max_h - tensor.shape[1]) 
                )
                for tensor, _, _, _ in items
            ], dim=0) 

            top_bottom_masks = torch.stack([ 
                F.pad(
                    torch.ones((1, tb.shape[2]), dtype=torch.bool), 
                    (0, max_w_tp - tb.shape[2], 0, 0)               
                )
                for _, tb, _, _ in items
            ], dim=0) 

            left_right_masks = torch.stack([  
                F.pad(
                    torch.ones((lr.shape[1], 1), dtype=torch.bool),  
                    (0, 0, 0, max_h_lr - lr.shape[1])                
                )
                for _, _, lr, _ in items
            ], dim=0) 
            
            gts    = torch.tensor([gt for _, _, _, gt in items], dtype=torch.long)

            sub_batches.append({
                "image": images,
                "top_bottom":top_bottom,
                "left_right":left_right,
                "image_mask": image_masks,           
                "top_bottom_mask": top_bottom_masks, 
                "left_right_mask": left_right_masks, 
                "gt":gts
            })

        return sub_batches
    
    return custom_collate_fn


class DocMDetectorDataPLModule(pl.LightningDataModule):
    """DataModule for G_theta bounding box quality training."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_size = self.config.train_batch_size
        self.train_dataset = None
        self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=make_custom_collate_fn(bin_size=self.config.bin_size),
        )

    def val_dataloader(self):
        return  DataLoader(
                self.val_dataset,
                batch_size=1,
                pin_memory=False,
                shuffle=False,
                drop_last=False,
                num_workers=1,
        )#no validation here, kept in case it is needed in the future









