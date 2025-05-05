# Pretrained PatchTST model for TSAD; anomaly score = prediction error
# If using this version, we should add 1 elements to the input sequence ie for each dataset end+1

import torch 
import torch.nn as nn
import lightning as L
from models.base_model import PatchTrADencoder
from models.metric import StreamAUC
    
class PredictorHead(nn.Module):
    def __init__(self, n_vars, patch_num,  d_model, head_dp=0.):
        super().__init__() 

        self.n_vars = n_vars
        dim = d_model*patch_num

        self.layers = nn.ModuleList([])
        for _ in range(n_vars):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim//2),
                    nn.Dropout(head_dp),
                    nn.ReLU(),
                    nn.Linear(dim//2, 1)
                )
            )

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches x d_model

        # Output:

        # out: bs x nvars x 1

        outs = []
        for i in range(self.n_vars):
            input = x[:, i, :, :]
            input = input.flatten(start_dim=1)
            out = self.layers[i](input)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs
    
class PatchTrAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config.ws // config.patch_len

        self.encoder = PatchTrADencoder(config)
        checkpoint_path = config.save_path + "_" + str(config.load_epoch) + ".ckpt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.requires_grad_(False if config.freeze_encoder else True)

        self.head = PredictorHead(config.in_dim, num_patches, config.d_model, config.head_dp if config.head_dp else 0)
        self.head.requires_grad_(True)
        
        self.cri = nn.MSELoss(reduction="none")

    def forward(self, x):
        input, target = x[:, :-1, :], x[:, -1, :]
        _, h = self.encoder(input)

        out = self.head(h)
        return target, out
    
    def get_loss(self, x, mode="train"):

        target, out = self.forward(x)
        out = out.squeeze(-1)
        error = self.cri(out, target)

        error = error.flatten(start_dim=1)
        error = error.mean(dim=1)
        
        return error
    
class PatchTradLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr

        self.auc = StreamAUC()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.model.get_loss(x, mode="train")
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        return self.model.get_loss(x, mode=mode)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")

        self.auc.update(errors, y)
    
    def on_test_epoch_end(self):
     
        auc = self.auc.compute()
        self.log("auc", auc, prog_bar=True)
        self.auc.reset()