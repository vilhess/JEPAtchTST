import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import TSDataset
from models.forecaster import JePatchTST
from utils import save_results

@hydra.main(version_base=None, config_path=f"../conf", config_name="config")
def main(cfg: DictConfig):

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    cfg.pretraining = None
    cfg = OmegaConf.merge(cfg.forecasting, cfg.encoder)

    if cfg.univariate:
        cfg.in_dim = 1

    for scratch, freeze_encoder in [(True, True), (False, True), (False, False)]:
        cfg.scratch = scratch
        cfg.freeze_encoder = freeze_encoder
        
        torch.manual_seed(0)

        wandb_logger = WandbLogger(project='ts-JEPA', name=f"Forecasting_{cfg.name}_{cfg.freeze_encoder}_{cfg.scratch}")
        
        trainset = TSDataset(
            path=cfg.path,
            seq_len=cfg.ws,
            target_len=cfg.target_len,
            train=True,
            univariate=cfg.univariate,
            target="OT"
        )
        testset = TSDataset(
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
                            accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False, callbacks=[EarlyStopping(monitor="val_l2loss", mode="min", patience=10)])
        trainer.fit(model=model, train_dataloaders=trainloader)

        results = trainer.test(model=model, dataloaders=testloader)
        total_loss = results[0]["mse"]

        ext=f"_univariate" if cfg.univariate else ""
        rev = "_revin" if cfg.revin else ""
        save_results(filename=f"results/mse_{cfg.size}.json", dataset=f"{cfg.name}{ext}", model=f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}{rev}", score=total_loss)

        wandb_logger.experiment.summary[f"test_mse"] = total_loss
        wandb.finish()

if __name__ == "__main__":
    main()