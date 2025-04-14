import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset import ETTDataset
from models.forecaster import JePatchTST
from utils import save_results

@hydra.main(version_base=None, config_path=f"../conf", config_name="config")
def main(cfg: DictConfig):

    torch.manual_seed(0)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    cfg.pretraining = None
    cfg = OmegaConf.merge(cfg.forecasting, cfg.encoder)

    wandb_logger = WandbLogger(project='ts-JEPA', name=f"Forecasting_{cfg.name}_{cfg.freeze_encoder}_{cfg.scratch}")
    
    trainset = ETTDataset(
        path=cfg.path,
        seq_len=cfg.ws,
        target_len=cfg.target_len,
        train=True,
        univariate=cfg.univariate,
        target="OT"
    )
    testset = ETTDataset(
        path=cfg.path,
        seq_len=cfg.ws,
        target_len=cfg.target_len,
        train=False,    
        univariate=cfg.univariate,
        target="OT"
    )
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=21)
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=21)

    model = JePatchTST(config=cfg)

    wandb_logger.config = cfg

    trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, 
                        accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False)
    trainer.fit(model=model, train_dataloaders=trainloader)

    model = model.model.to(DEVICE)
    cri  = nn.MSELoss()
    total_loss = []

    for batch in tqdm(testloader):
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE).transpose(1, 2)
        with torch.no_grad():
            pred = model(x)
            loss = cri(pred, y)
            total_loss.append(loss.item())  
    total_loss = np.average(total_loss)
    print(f"Test Loss: {total_loss}")
    save_results(filename="results/mse.json", dataset=cfg.name, model=f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}", score=total_loss)

    wandb_logger.experiment.summary[f"test_mse"] = total_loss
    wandb.finish()

if __name__ == "__main__":
    main()