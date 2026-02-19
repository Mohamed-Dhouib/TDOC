
import math
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from model import DocMDetectorConfig, DocMDetector
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.functional as F
import os
import gc
from torch.nn.functional import pad


def _worker_init_threads(_):
    """Disable OpenCV multi-threading in dataloader workers."""
    import cv2, os
    try:
        cv2.setNumThreads(0)       
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


class DocMDetectorPLModule(pl.LightningModule):
    """Training module for F_theta using contrastive loss (Algorithm 1, Section 3.1)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.temperature= self.config.temperature
        self.accumulate_grad_batches=self.config.accumulate_grad_batches


        self.model = DocMDetector(
                config=DocMDetectorConfig(
                     self.config.pretrained_model_name_or_path,
                )
            )
            
    def training_step(self, batch, batch_idx):
        opt   = self.optimizers()
        sched = self.lr_schedulers()
        grad_accumulation=self.accumulate_grad_batches
        backprop_now = ((batch_idx+1)% grad_accumulation) == 0

        if batch_idx==0 or not hasattr(self, 'loss'):
            self.loss=[]

        PIXEL_BUDGET = self.config.pixel_budget
        total_anchors = sum(el["anchor"].shape[0] for el in batch)
        sum_loss     = 0.0

        for index_batch,el in enumerate(batch):
            anchors, positives, negatives, is_blank_pos, is_blank_neg= el["anchor"], el["positive"], el["negatives"], el["is_blank_pos"], el["is_blank_neg"]
            G, C, H, W = anchors.shape
            max_chunk = max(1, PIXEL_BUDGET // (H*W))

            for start in range(0, G, max_chunk):
                end = min(G, start + max_chunk)
                a, p, n, bp, bn = anchors[start:end], positives[start:end], negatives[start:end], is_blank_pos[start:end], is_blank_neg[start:end] 
                g = end - start


                a_emb = self.model.encoder(a)
                p_emb = self.model.encoder(p)
                n_flat = n.view(g * n.size(1), C, H, W)
                n_emb  = self.model.encoder(n_flat).view(g, n.size(1), -1)
                
                d2 = a_emb.size(1) // 2

                a0, a1 = a_emb[:, :d2], a_emb[:, d2:]
                p0, p1 = p_emb[:, :d2], p_emb[:, d2:]
                n0, n1 = n_emb[:, :, :d2], n_emb[:, :, d2:]          

                bp = bp.to(a_emb.dtype).unsqueeze(1)                
                bn = bn.to(a_emb.dtype)                          

                pos_sim = (a0 * p0).sum(dim=1, keepdim=True)         
                pos_sim.mul_(0.5 + 0.5 * bp)                      
                tmp = (a1 * p1).sum(dim=1, keepdim=True)            
                pos_sim.add_(tmp.mul_(0.5 - 0.5 * bp))             
                del tmp

                neg_sim = torch.matmul(n0, a0.unsqueeze(-1)).squeeze(-1)  
                neg_sim.mul_(0.5 + 0.5 * bn)                             
                tmp = torch.matmul(n1, a1.unsqueeze(-1)).squeeze(-1)    
                neg_sim.add_(tmp.mul_(0.5 - 0.5 * bn))                  
                del tmp
                                

                logits  = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
                labels  = torch.zeros(g, dtype=torch.long, device=logits.device)

                chunk_loss = F.cross_entropy(logits, labels, reduction="sum")
                
                sum_loss += chunk_loss.item()

                loss_to_backprop = chunk_loss / (total_anchors*grad_accumulation)

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
                

        avg_loss = sum_loss / total_anchors

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


def custom_collate_fn(batch):
    """
    1) flatten nested lists of dicts
    2) assign each sample to a (h_bin, w_bin) where h_bin=ceil(H/bin_size)*bin_size, w_bin=ceil(W/bin_size)*bin_size
    3) within each bin, pad *to that bin’s actual* max H and W
    4) return a list of sub-batches
    """
    flat = [item for sub in batch for item in sub]
    bin_size=1

    bins = {}
    for item in flat:
        C, H, W = item["anchor"].shape
        h_bin = math.ceil(H / bin_size) * bin_size
        w_bin = math.ceil(W / bin_size) * bin_size
        bins.setdefault((h_bin, w_bin), []).append(item)

    sub_batches = []
    for items in bins.values():
        max_h = max(it["anchor"].shape[1] for it in items)
        max_w = max(it["anchor"].shape[2] for it in items)

        def pad_to_target(img: torch.Tensor):
            _, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            return pad(img, (0, pad_w, 0, pad_h))

        anchors = torch.stack([pad_to_target(it["anchor"]) for it in items], dim=0)
        positives = torch.stack([pad_to_target(it["positive"]) for it in items], dim=0)
        negatives = torch.stack([
            torch.stack([pad_to_target(n) for n in it["negatives"]], dim=0)
            for it in items
        ], dim=0)
        is_blank_neg=  torch.stack([it["is_blank_neg"] for it in items], dim=0)
        is_blank_pos=  torch.stack([it["is_blank_pos"] for it in items], dim=0)

        sub_batches.append({
            "anchor":    anchors,
            "positive":  positives,
            "negatives": negatives,
            "is_blank_pos": is_blank_pos,
            "is_blank_neg": is_blank_neg,
        })

    return sub_batches







class DocMDetectorDataPLModule(pl.LightningDataModule):
    """DataModule for F_theta contrastive training (Algorithm 1)."""

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
            collate_fn=custom_collate_fn,
            worker_init_fn=_worker_init_threads,
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









