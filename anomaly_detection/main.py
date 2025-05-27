import sys
sys.path.append("..")

import torch
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

from models.anomaly_detector import JEPAtchTrADLit
from utils import get_loaders, save_results

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(version_base=None, config_path=f"../conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    cfg.pretraining = None
    
    cfg = OmegaConf.merge(cfg.anomaly_detection, cfg.encoder)
    dataset = cfg.name

    loaders = get_loaders(dataset, cfg)

    wandb_logger = WandbLogger(project='ts-JEPA', name=f"{dataset}")
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        cfg["len_loader"] = len(trainloader) # Useful for some lr scheduler
        wandb_logger.config = cfg
        
        LitModel = JEPAtchTrADLit(cfg)
        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, precision="bf16-mixed", accelerator="gpu", devices=1, strategy="auto")

        trainer.fit(model=LitModel, train_dataloaders=trainloader)
        
        results = trainer.test(model=LitModel, dataloaders=testloader)
        auc = results[0]["auc"]

        print(f"AUC: {auc}")

        aucs.append(auc)
        wandb_logger.experiment.summary[f"auc_subset_{i+1}/{len(loaders)}"] = auc

        ### Free memory after each subset
        LitModel.to("cpu")
        del LitModel
        del trainloader, testloader
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        del trainer
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        ###
        
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    save_results(filename=f"results/aucs.json", dataset=dataset, model=f"jepatchtrad", score=round(final_auc, 4))
    wandb_logger.experiment.summary[f"final_auc"] = final_auc

    wandb.finish()

if __name__ == "__main__":
    main()