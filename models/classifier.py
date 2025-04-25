import sys
sys.path.append("..")

import torch 
import torch.nn as nn
import lightning as L
from models.base_model import PatchTrADencoder
    
class ClassifierHead(nn.Module):
    def __init__(self, n_vars, patch_num,  d_model, n_classes, head_dp=0.):
        super().__init__() 

        dim = n_vars*d_model*patch_num

        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(head_dp),
            nn.Linear(dim, n_classes), 
            nn.Sigmoid()
        )

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches x d_model

        # Output:

        # out: bs x n_class

        outs = self.layers(x)
        return outs
    
class PatchTrAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config.ws // config.patch_len

        self.encoder = PatchTrADencoder(config)

        if not config.scratch:
            checkpoint_path = "../" + config.save_path + "_" + str(config.load_epoch) + ".ckpt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.encoder.load_state_dict(checkpoint)
            self.encoder.requires_grad_(False if config.freeze_encoder else True)

        self.head = ClassifierHead(
            n_vars=config.in_dim,
            patch_num=num_patches,
            d_model=config.d_model,
            n_classes=config.n_classes,
            head_dp=config.head_dp
        )
        self.head.requires_grad_(True)

    def forward(self, x):
        _, h = self.encoder(x)

        prediction = self.head(h)
        return prediction
    
class JePatchTST(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer