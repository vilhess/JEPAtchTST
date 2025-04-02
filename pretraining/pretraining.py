import sys
sys.path.append("..")

import torch
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from models.mask import MaskCollator 
from models.base_model import LitJEPA
#from dataset.pretrain_ds import create_dataloader
from dataset.uts import create_dataloader

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:

    OmegaConf.set_struct(cfg, False)
    wandb_logger = WandbLogger(project='TS-JEPA', name='PreTraining', config=dict(cfg))

    print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.merge(cfg.pretraining, cfg.base_encoder)

    collator = MaskCollator(ratio=cfg.mask_ratio_range, window_size=cfg.ws, patch_len=cfg.patch_len)
    #trainloader = create_dataloader(ws=cfg.ws, batch_size=cfg.batch_size, data_dir=cfg.data_dir, collator=collator)
    trainloader = create_dataloader(subset_name='UTSD-1G', window_size=cfg.ws, batch_size=cfg.batch_size, collator=collator)
    cfg["len_loader"] = len(trainloader)
    
    model = LitJEPA(cfg)
    trainer = L.Trainer(max_epochs=cfg.epochs, enable_checkpointing=True, log_every_n_steps=50, fast_dev_run=False, logger=wandb_logger, precision="bf16-mixed", accelerator="gpu", devices=1, strategy="auto")
    trainer.fit(model=model, train_dataloaders=trainloader)

    wandb.finish()

if __name__ == '__main__':
    main()