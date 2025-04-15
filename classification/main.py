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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import SignalDataset, signal_type_to_label
from models.classifier import JePatchTST
from utils import save_results

@hydra.main(version_base=None, config_path=f"../conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        

        torch.manual_seed(0)

        wandb_logger = WandbLogger(project='ts-JEPA', name=f"Classification_{cfg.name}_{cfg.freeze_encoder}_{cfg.scratch}")
        
        trainset = SignalDataset(mode='train')
        testset = SignalDataset(mode='test')

        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=21)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=21)

        model = JePatchTST(config=cfg)

        wandb_logger.config = cfg

        trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, 
                            accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False)
        trainer.fit(model=model, train_dataloaders=trainloader)

        model = model.model.to(DEVICE)
        model.eval()

        all_preds = []
        all_targets = []

        for batch in tqdm(testloader):
            x, y = batch
            x = x.to(DEVICE)
            with torch.no_grad():
                logits = model(x)
            preds = torch.argmax(logits, dim=1).detach().cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.tolist())

        # Compute accuracy
        correct = sum(p == t for p, t in zip(all_preds, all_targets))
        accuracy = correct / len(all_targets)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        label_to_signal_type = {v: k for k, v in signal_type_to_label.items()}
        class_names = [label_to_signal_type[i] for i in range(len(label_to_signal_type))]

        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix")
        plt.savefig(f"results/confusion_matrix_{cfg.freeze_encoder}_{cfg.scratch}.png")
        plt.close()

        save_results(filename="results/accs.json", dataset=cfg.name, model=f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}", score=accuracy)

        wandb_logger.experiment.summary[f"test_accuracy"] = accuracy
        wandb.finish()

if __name__ == "__main__":
    main()