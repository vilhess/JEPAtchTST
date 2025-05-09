import sys
sys.path.append("../../")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import load_concat_datasets
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
        
        trainset, testset, signal_type_to_label = load_concat_datasets(seq_len=cfg.ws)

        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, persistent_workers=True)

        cfg.n_classes=len(signal_type_to_label)
        cfg.epochs=100

        model = JePatchTST(config=cfg)

        wandb_logger.config = cfg

        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, 
                            accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False, callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=10)])
        trainer.fit(model=model, train_dataloaders=trainloader)

        results = trainer.test(model=model, dataloaders=testloader)
        accuracy = results[0]["acc"]
        print(f"Accuracy: {accuracy * 100:.2f}%")

        model_name = f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}"
        save_results(filename=f"results/{cfg.size}/accs.json", dataset=cfg.name, model=model_name, score=accuracy)

        wandb_logger.experiment.summary[f"test_accuracy"] = accuracy
        wandb.finish()

if __name__ == "__main__":
    main()