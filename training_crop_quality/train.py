import argparse
import os
from os.path import basename
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
from util import DocMDetectorDataset
from lightning_module import DocMDetectorDataPLModule, DocMDetectorPLModule
from pytorch_lightning.callbacks import TQDMProgressBar
import shutil
import datetime

cols, rows = shutil.get_terminal_size(fallback=(120, 30))
term_width = cols 


class CustomTqdm(TQDMProgressBar):
    """Custom progress bar with fixed terminal width."""

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

class CheckpointEveryNSteps(pl.Callback):
    """Save checkpoint periodically during training."""

    def __init__(
        self,
        steps_per_checkpoint=100,
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
    """Save training configuration to YAML file."""
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    """Train G_theta bounding box quality network (Section 3.2)."""

    model_module = DocMDetectorPLModule(config)
    data_module = DocMDetectorDataPLModule(config)

    datasets = {}
    for split in ["train"]:
        datasets[split]= DocMDetectorDataset(
                dataset_name_or_path=config.dataset_name_or_path,
            )

    data_module.train_dataset = datasets["train"]
    data_module.val_dataset = None

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    callbacks_list=[lr_callback,CustomTqdm()]
    if config.check_point_every_n_step :
        callbacks_list.append(CheckpointEveryNSteps(steps_per_checkpoint=config.steps_per_checkpoint))

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
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        precision=config.precision,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks_list,
        
    )
    

    if config.compile_model :

        torch._dynamo.config.optimize_ddp = False
        model_module.model.encoder =  torch.compile(model_module.model.encoder,dynamic=True, mode="max-autotune-no-cudagraphs")






    trainer.fit(model_module, data_module )


if __name__ == "__main__":
    print(torch._dynamo.list_backends())
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

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    train(config)
