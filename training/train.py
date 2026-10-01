import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLOSC_NTHREADS'] = '1'
os.environ["PL_FAULT_TOLERANT_TRAINING"] = "0"   # no autosave blobs on crash/timeout
os.environ["LIGHTNING_FAULT_TOLERANT"] = "0"     # Fabric variant; harmless if unused

import argparse
from os.path import basename
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
from util import DocMDetectorDataset
from lightning_module import DocMDetectorDataPLModule, DocMDetectorPLModule
from pytorch_lightning.callbacks import TQDMProgressBar
import shutil
import datetime
from pytorch_lightning.strategies import DDPStrategy
import ast
import json
import sys
import traceback
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_rank_caches() -> tuple[str, str]:
    """
    Build per-job/per-node/per-rank cache directories for TorchInductor and Triton.
    - No asserts: uses safe fallbacks when env vars are missing.
    - Creates directories if needed.
    - Exports TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR for this process.
    - Prints the final resolved paths.
    Returns: (final_inductor_path, final_triton_path)
    """

    home = Path.home()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    base_inductor = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", xdg_cache / "torch" / "inductor"))
    base_triton   = Path(os.environ.get("TRITON_CACHE_DIR", xdg_cache / "triton"))

    node = os.environ.get("SLURM_NODEID") or os.environ.get("NODE_RANK") or "0"
    rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or "0"
    )

    sub = f"node{node}_rank{rank}"
    final_inductor = (base_inductor / sub).resolve()
    final_triton   = (base_triton / sub).resolve()

    final_inductor.mkdir(parents=True, exist_ok=True)
    final_triton.mkdir(parents=True, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(final_inductor)
    os.environ["TRITON_CACHE_DIR"] = str(final_triton)

    print(f"TORCHINDUCTOR_CACHE_DIR={final_inductor}")
    print(f"TRITON_CACHE_DIR={final_triton}")

    return str(final_inductor), str(final_triton)


cols, rows = shutil.get_terminal_size(fallback=(120, 30))
term_width = cols 



class CustomTqdm(TQDMProgressBar):
    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.ncols = term_width
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ncols = term_width
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ncols = term_width
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.ncols = term_width
        return bar

class CheckpointAtTrainEnd(pl.Callback):
    def __init__(self, filename="periodic.pth"):
        self.filename = filename

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        ckpt_dir = trainer.checkpoint_callback.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, self.filename)
        print(f"saving on {ckpt_path}")
        checkpoint = trainer.model.state_dict()
        torch.save(checkpoint, ckpt_path)

class CheckpointEveryNSteps(pl.Callback):


    def __init__(
        self,
        steps_per_checkpoint=1000,
    ):

        self.steps_per_checkpoint = steps_per_checkpoint

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, _, a, b, batch_idx):
        if (batch_idx) % self.steps_per_checkpoint== 0 and batch_idx!=0 :
            print("saving peridically")
            filename = "periodic.pth"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            os.makedirs(trainer.checkpoint_callback.dirpath,exist_ok=True)
            print("saving on"+str(ckpt_path))
            checkpoint=trainer.model.state_dict()
            torch.save(checkpoint, ckpt_path)




@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):

    model_module = DocMDetectorPLModule(config)
    data_module = DocMDetectorDataPLModule(config)

    data_module.train_dataset = DocMDetectorDataset(
        dataset_name_or_path=config.dataset_name_or_path,
        seg_model=model_module.model,
        split="train",
        config=config,
    )

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    ckpt_dir = Path(config.result_path) / config.exp_name / config.exp_version

    noop_ckpt = ModelCheckpoint(
    dirpath=str(ckpt_dir / "checkpoints"),
    save_top_k=0,     # save no "best" checkpoints
    save_last=False,  # don't save "last"
)

    class UnusedParametersDetector(pl.Callback):
        """Callback to detect and print unused parameters in the model."""
        
        def __init__(self):
            self.hooks = []
            self.param_used = {}
        
        @rank_zero_only
        def on_train_start(self, trainer: pl.Trainer, pl_module):
            print("\n" + "="*80)
            print("DETECTING UNUSED PARAMETERS")
            print("="*80)
            
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.param_used[name] = False
                    
                    def make_hook(param_name):
                        def hook(grad):
                            self.param_used[param_name] = True
                            return grad
                        return hook
                    
                    handle = param.register_hook(make_hook(name))
                    self.hooks.append(handle)
        
        @rank_zero_only
        def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx == 0:
                unused_params = [name for name, used in self.param_used.items() if not used]
                
                if unused_params:
                    print("\n" + "="*80)
                    print(f"WARNING: Found {len(unused_params)} unused parameters:")
                    print("="*80)
                    
                    module_groups = {}
                    for param_name in unused_params:
                        module_name = '.'.join(param_name.split('.')[:-1])
                        if module_name not in module_groups:
                            module_groups[module_name] = []
                        module_groups[module_name].append(param_name.split('.')[-1])
                    
                    for module_name, params in sorted(module_groups.items()):
                        print(f"\nModule: {module_name}")
                        for param in params:
                            print(f"  - {param}")
                    
                    print("\n" + "="*80)
                    print(f"Total unused parameters: {len(unused_params)}")
                    print("="*80 + "\n")
                else:
                    print("\n" + "="*80)
                    print("✓ All parameters are being used!")
                    print("="*80 + "\n")
                
                for handle in self.hooks:
                    handle.remove()
                self.hooks = []

    callbacks_list=[noop_ckpt, lr_callback,CustomTqdm(),CheckpointAtTrainEnd(),UnusedParametersDetector()] #checkpoint_every_epoch


    if config.check_point_every_n_step :
        callbacks_list.append(CheckpointEveryNSteps(config.steps_per_checkpoint))

    try :
        nodes= int(os.getenv('SLURM_JOB_NUM_NODES'))
        if nodes<1 :
            nodes=1
    except :
        nodes=1

    print(f"Training will run on {nodes} nodes with {torch.cuda.device_count()} gpus per node.")

    trainer = pl.Trainer(
        num_nodes=nodes,
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False,timeout=timedelta(minutes=60)), #altered timeout fot models to get time to compile
        accelerator="auto",
        max_epochs=config.max_epochs,
        default_root_dir=ckpt_dir,
        precision=config.precision,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks_list,
        accumulate_grad_batches=config.accumulate_grad_batches,
        
    )
    

    if config.compile_model:
        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.recompile_limit = 512
        torch._dynamo.config.cache_size_limit = 512
        if config.model_name=="ascformer" or "ffdn" in config.model_name:
            print("compiling without cuda graphs")
            model_module.model.encoder =  torch.compile(model_module.model.encoder, mode="max-autotune-no-cudagraphs", dynamic=False )
        else:
            model_module.model.encoder =  torch.compile(model_module.model.encoder, mode="max-autotune")


    trainer.fit(model_module, data_module, ckpt_path=config.get("resume_from_checkpoint_path",None) )


if __name__ == "__main__":
    print("using new script")
    setup_rank_caches()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()


    config = Config(args.config)
    config.argv_update(left_argv)
    if config.precision!=32 :
        torch.set_float32_matmul_precision("medium")

    config.exp_name = basename(args.config).split(".")[0]
    
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version
    
    config.save_path_=os.path.join(config.result_path,config.exp_name,config.exp_version)

    
    dataset_name_or_path = ast.literal_eval(os.environ.get("DATASET_NAME_OR_PATH",str(config["dataset_name_or_path"])))
    print(f"Training on the following dataset input {dataset_name_or_path}")
    config["dataset_name_or_path"]=dataset_name_or_path

    model_name = os.environ.get("MODEL_NAME",str(config["model_name"]))
    print(f"Training the following model: {model_name}")
    config["model_name"]=model_name


    pretrained_model_name_or_path=os.environ.get("PRETRAINED_PATH",config["pretrained_model_name_or_path"])
    config["pretrained_model_name_or_path"]=pretrained_model_name_or_path
    print(f"pretrained_model_name_or_path is set to {pretrained_model_name_or_path}")
  
    
    config.finetuning=config.get("finetuning",False)
    
    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    print(config)
    pl.seed_everything(config.seed, workers=True)
    train(config)
