import sys
sys.path.append("../../")

import torch
from torch.utils.data import DataLoader
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset import SignalDataset
from models.classifier import JePatchTST
from classification.resnet import ResNetLit
from aeon_training import training_module, test_module
from utils import save_results

@hydra.main(version_base=None, config_path=f"../../conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    cfg.pretraining = None
    cfg = OmegaConf.merge(cfg.classification, cfg.encoder)

    for scratch, freeze_encoder in [('HIVECOTEV2', None)]: # (True, True), (False, True), (False, False), ('ResNet', None), ('KNN_DTW', None), 

        if type(scratch) is bool:
            cfg.scratch = scratch
            cfg.freeze_encoder = freeze_encoder
            project_name = f"Classification_{cfg.name}_{cfg.freeze_encoder}_{cfg.scratch}"

        else:
            project_name = f"Classification_{scratch}"
        
        torch.manual_seed(0)

        wandb_logger = WandbLogger(project='ts-JEPA', name=project_name)
        
        trainset = SignalDataset(mode='train', size=cfg.ws)
        testset = SignalDataset(mode='test', size=cfg.ws)

        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

        if type(scratch) is bool:
            model = JePatchTST(config=cfg)
        elif scratch=="ResNet":
            model = ResNetLit(config=cfg)
        elif scratch in ["KNN_DTW", "HIVECOTEV2"]:
            model = scratch

        wandb_logger.config = cfg

        if scratch in ["KNN_DTW", "HIVECOTEV2"]:
            model = training_module(model, trainloader)
            all_preds, all_targets = test_module(model, testloader)
        
        else:

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

        if type(scratch) is bool:
            model_name = f"JePatchTST_{cfg.freeze_encoder}_{cfg.scratch}"
        else:
            model_name = scratch
        save_results(filename=f"results/{cfg.size}/accs.json", dataset=cfg.name, model=model_name, score=accuracy)

        wandb_logger.experiment.summary[f"test_accuracy"] = accuracy
        wandb.finish()

if __name__ == "__main__":
    main()