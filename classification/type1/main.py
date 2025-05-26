import sys
sys.path.append("../../")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import SignalDataset
from models.classifier import JePatchTST
from utils import save_results

@hydra.main(version_base=None, config_path=f"../../conf", config_name="config")
def main(cfg: DictConfig):

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    cfg.pretraining = None
    cfg = OmegaConf.merge(cfg.classification, cfg.encoder)

    for scratch, freeze_encoder in [(True, True), (False, True), (False, False)]: 
        
        cfg.scratch = scratch
        cfg.freeze_encoder = freeze_encoder
        project_name = f"Classification_{cfg.name}_{cfg.freeze_encoder}_{cfg.scratch}"

        torch.manual_seed(0)

        wandb_logger = WandbLogger(project='ts-JEPA', name=project_name)
        
        trainset = SignalDataset(mode='train', size=cfg.ws)
        valset = SignalDataset(mode='val', size=cfg.ws)
        testset = SignalDataset(mode='test', size=cfg.ws)

        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
        valloader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

        model = JePatchTST(config=cfg)

        wandb_logger.config = cfg

        early_stop_callback = EarlyStopping(monitor="val_acc", mode="max", patience=50)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=False,
            filename="best-checkpoint"
        )

        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=True, log_every_n_steps=1, 
                            accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)

        best_model_path = checkpoint_callback.best_model_path
        best_model = JePatchTST.load_from_checkpoint(best_model_path, config=cfg)
        results = trainer.test(model=best_model, dataloaders=testloader)
        accuracy = results[0]["acc"]

        print(f"Accuracy: {accuracy * 100:.2f}%")

        model_name = f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}"
        save_results(filename=f"results/accs.json", dataset=cfg.name, model=model_name, score=accuracy)

        wandb_logger.experiment.summary[f"test_accuracy"] = accuracy
        wandb.finish()

if __name__ == "__main__":
    main()