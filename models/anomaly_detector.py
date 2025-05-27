import torch 
import torch.nn as nn
import lightning as L
from models.base_model import JEPAtchTSTEncoder
from models.metric import StreamAUC
    
class Head(nn.Module):
    def __init__(self, n_vars, patch_len, patch_num,  d_model, head_dp=0.):
        super().__init__() 

        self.n_vars = n_vars

        self.layers = nn.ModuleList([])
        for _ in range(n_vars):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(d_model, patch_len),
                    nn.Dropout(head_dp),
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
            out = self.layers[i](input)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs
    
class JEPAtchTrAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config["ws"] // config["patch_len"]

        if config["load_hub"]:
            print("Loading pretrained JEPAtchTST from Hugging Face Hub")
            self.encoder = JEPAtchTSTEncoder.from_pretrained("vilhess/JEPAtchTST")
        else:
            print("Loading JEPAtchTST from local checkpoint")
            self.encoder = JEPAtchTSTEncoder(config)
            checkpoint_path = config["save_path"]
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.encoder.load_state_dict(checkpoint)

        self.encoder.requires_grad_(False if config["freeze_encoder"] else True)

        self.head = Head(config["in_dim"], config["patch_len"], num_patches, config["d_model"], config["head_dp"] if config["head_dp"] else 0)
        self.head.requires_grad_(True)

    def forward(self, x):
        patched, h = self.encoder(x)

        out = self.head(h)
        return patched, out
    
    def get_loss(self, x, mode="train"):

        inp, out = self.forward(x)

        if mode=="train":
            error = ((out - inp)**2).flatten(start_dim=1).mean(dim=(1))
            
        elif mode=="test":
            inp = inp[:, :, -1, :]
            out = out[:, :, -1, :]
            error = ((out - inp)**2).flatten(start_dim=1).mean(dim=(1))

        return error
    
class JEPAtchTrADLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = JEPAtchTrAD(config)
        self.lr = config["lr"]

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