import torch 
import torch.nn as nn
import lightning as L
from models.base_model import PatchTrADencoder
    
class PredictorHead(nn.Module):
    def __init__(self, n_vars, patch_num,  d_model, target_len, head_dp=0.):
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
                    nn.Linear(dim//2, target_len)
                )
            )

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches x d_model

        # Output:

        # out: bs x nvars x num_patches x patch_len

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

        if not config.scratch:
            checkpoint_path = config.save_path + "_" + str(config.load_epoch) + ".ckpt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.encoder.load_state_dict(checkpoint)
            self.encoder.requires_grad_(False if config.freeze_encoder else True)

        self.head = PredictorHead(
            n_vars=config.in_dim,
            patch_num=num_patches,
            d_model=config.d_model,
            target_len=config.target_len,
            head_dp=config.head_dp
        )
        self.head.requires_grad_(True)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        _, h = self.encoder(x)

        prediction = self.head(h)
        return prediction
    
class JePatchTST(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        y = y.transpose(1, 2)
        loss = self.model.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer