import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import TSDataset
from models.forecaster import JePatchTSTLit
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
            mode="train",
            univariate=cfg.univariate,
            target="OT"
        )
        valset = TSDataset(
            path=cfg.path,
            seq_len=cfg.ws,
            target_len=cfg.target_len,
            mode="val",
            univariate=cfg.univariate,
            target="OT"
        )
        testset = TSDataset(
            path=cfg.path,
            seq_len=cfg.ws,
            target_len=cfg.target_len,
            mode="test",   
            univariate=cfg.univariate,
            target="OT"
        )
        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=21)
        valloader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=21)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=21)

        model = JePatchTSTLit(config=cfg)

        wandb_logger.config = cfg

        early_stop_callback = EarlyStopping(monitor="val_l2loss", mode="min", patience=10)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_l2loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            filename="best-checkpoint"
        )

        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=True, log_every_n_steps=1, 
                            accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)

        best_model_path = checkpoint_callback.best_model_path
        best_model = JePatchTSTLit.load_from_checkpoint(best_model_path, config=cfg)
        results = trainer.test(model=best_model, dataloaders=testloader)
        total_loss = results[0]["l2loss"]

        ext=f"_univariate" if cfg.univariate else ""
        rev = "_revin" if cfg.revin else ""
        save_results(filename=f"results/mse.json", dataset=f"{cfg.name}{ext}", model=f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}{rev}", score=total_loss)

        wandb_logger.experiment.summary[f"test_mse"] = total_loss
        wandb.finish()

if __name__ == "__main__":
    main()