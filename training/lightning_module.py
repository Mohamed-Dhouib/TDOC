import math
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from model import DocMDetectorConfig, DocMDetector
import torch.nn.functional as F
import torch.nn as nn
import os

def _worker_init_threads(_):
    import os, cv2, torch
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['BLOSC_NTHREADS'] = '1'


    torch.set_num_threads(1)
    



class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation."""
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
      
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )


class DocMDetectorPLModule(pl.LightningModule):
    """Lightning module for document manipulation detection training."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.register_buffer(
            "seg_loss_weight",
            torch.tensor([1.0, config.loss_pos_weight], dtype=torch.float32)
        )
        self.register_buffer(
            "class_loss_weight",
            torch.tensor([1.0, 1.0], dtype=torch.float32)
        )

        self.Focal_fn=FocalLoss(gamma=2.,weight=self.seg_loss_weight)
        self.Focal_fn_class=FocalLoss(gamma=2.,weight=self.class_loss_weight)
     
        self.model = DocMDetector(
            config=DocMDetectorConfig(
                pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                data_aug_config=self.config.data_aug_config,
                model_name=self.config.model_name,
            )
        )
      
    def training_step(self, batch, batch_idx):
        if batch_idx==0 or not hasattr(self, 'loss'):
            self.loss=[]
            self.loss_class=[]

        image_tensors, masks, dcts, dwts, ela, qt = batch
        
        pred,output_class=self.model.encoder(image_tensors,dcts,dwts,ela,qt)
 
        classification_gt = torch.any(input=(masks > 0).flatten(start_dim=1),dim=1).long()


        masks=masks.squeeze(1).long() 

        if self.config.model_name=="psccnet":
            seg_map, seg_map_1, seg_map_2, seg_map_3=pred

            loss_0=self.Focal_fn(seg_map,masks)
            loss_1=self.Focal_fn(seg_map_1,masks)
            loss_2=self.Focal_fn(seg_map_2,masks)
            loss_3=self.Focal_fn(seg_map_3,masks)
            loss= 0.65*loss_0+ 0.2*loss_1 +0.1*loss_2+ 0.05*loss_3

        else:
            loss=self.Focal_fn(pred,masks)

        loss_class=self.Focal_fn_class(output_class,classification_gt)

        final_loss=loss+loss_class*self.config.class_coef


        self.loss.append(loss.item())
        self.loss_class.append(loss_class.item())
      
        if len(self.loss)>5000:
            self.loss=self.loss[1:]  
            self.loss_class=self.loss_class[1:] 
       
        if (batch_idx) % 500 == 0:
            self.log_dict({"l_seg": sum(self.loss)/len(self.loss),"l_c" :sum(self.loss_class)/len(self.loss_class)}, sync_dist=True , prog_bar=True)
            if self.global_rank == 0:
                cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return final_loss

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
        
        param_map = dict(self.model.named_parameters())
        frozen_param_names = set()

        #keep DCT embedding frozen as in the original FPH
        if self.config.model_name == "DTD" or self.config.model_name == "ffdn":
            dtd_fph_params = [name for name in param_map if name.endswith("fph.obembed.weight")]
            if dtd_fph_params:
                for name in dtd_fph_params:
                    target_param = param_map[name]
                    assert not target_param.requires_grad, (
                        f"Expected {name} to be frozen by default, but it is trainable."
                    )
                    frozen_param_names.add(name)
                print(f"[FPH] Found {len(dtd_fph_params)} fph.obembed.weight param(s); kept frozen.")

        unfrozen_params = []
        for name, param in param_map.items():
            if name in frozen_param_names:
                param.requires_grad = False
            else:
                if not param.requires_grad:
                    unfrozen_params.append(name)
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for _, p in self.model.named_parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=0.01
        )

        try:
            min_lr = self.config.min_lr
        except KeyError:
            min_lr = 0.0  
            
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps, min_lr=min_lr),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps, min_lr: float = 0.0):
        """Cosine scheduler that clamps the lr to a minimum absolute value.

        Returns a LambdaLR where the multiplier is clamped so that
        optimizer_base_lr * multiplier >= min_lr.
        """
        # capture base lr from optimizer to compute minimum multiplier
        base_lr = float(optimizer.param_groups[0].get('lr', 1.0))
        min_factor = 0.0
        if base_lr > 0.0:
            min_factor = float(min_lr) / base_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return max(current_step / max(1, warmup_steps),min_factor)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, [lr_lambda])

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items




class DocMDetectorDataPLModule(pl.LightningDataModule):
    """Lightning data module for document manipulation detection."""
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
            persistent_workers=self.config.get("persistent_workers", False),
            worker_init_fn=_worker_init_threads,
            )
        

        

