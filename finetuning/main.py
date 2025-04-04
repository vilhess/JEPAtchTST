import sys
sys.path.append("..")

import torch
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import gc

from models.ft_model import PatchTradLit
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
    
    cfg = OmegaConf.merge(cfg.finetuning, cfg.encoder)
    dataset = cfg.name

    loaders = get_loaders(dataset, cfg)

    wandb_logger = WandbLogger(project='ts-JEPA', name=f"{dataset}")
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        cfg["len_loader"] = len(trainloader) # Useful for some lr scheduler
        wandb_logger.config = cfg
        
        LitModel = PatchTradLit(cfg)
        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, precision="bf16-mixed", accelerator="gpu", devices=1, strategy="auto")
        #trainer = L.Trainer(max_epochs=1, logger=wandb_logger, enable_checkpointing=False, fast_dev_run=True)

        trainer.fit(model=LitModel, train_dataloaders=trainloader)
        
        test_errors = []
        test_labels = []
        
        LitModel = LitModel.to(DEVICE)
        LitModel.eval()

        with torch.no_grad():
            pbar = tqdm(testloader, desc="Detection Phase")
            for x, anomaly in pbar:
                x = x.to(DEVICE)
                errors = LitModel.get_loss(x, mode="test")

                test_labels.append(anomaly)
                test_errors.append(errors)
                del x

        test_errors = torch.cat(test_errors).detach().cpu()
        test_labels = torch.cat(test_labels).detach().cpu()

        auc = roc_auc_score(test_labels, test_errors)
        print(f"AUC: {auc}")

        if dataset in ["nyc_taxi", "ec2_request_latency_system_failure"]:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 6))
            plt.plot( test_errors.numpy(), label="Test Errors")
            for idx in range(len(test_labels)):
                if test_labels[idx] == 1:
                    plt.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.3) 
            plt.title(f"Test Errors for {dataset} subset {i+1}/{len(loaders)}: auc = {auc:.4f}")
            plt.savefig(f"plots/{dataset}_{i+1}.png")
            plt.close()

        aucs.append(auc)
        wandb_logger.experiment.summary[f"auc_subset_{i+1}/{len(loaders)}"] = auc

        ### Free memory after each subset
        LitModel.to("cpu")
        del LitModel
        del test_errors, test_labels, trainloader, testloader
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        del trainer
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        ###
        
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    save_results(filename="results/aucs.json", dataset=dataset, score=round(final_auc, 4))
    wandb_logger.experiment.summary[f"final_auc"] = final_auc

    wandb.finish()

if __name__ == "__main__":
    main()