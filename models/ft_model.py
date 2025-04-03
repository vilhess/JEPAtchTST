import torch 
import torch.nn as nn
import lightning as L
from models.base_model import PatchTrADencoder
    
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

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
class PatchTrAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config.ws // config.patch_len

        self.encoder = PatchTrADencoder(config)
        checkpoint_path = config.save_path + "_" + str(config.load_epoch) + ".ckpt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.encoder.load_state_dict(checkpoint)
        self.encoder.requires_grad_(False if config.freeze_encoder else True)

        self.head = Head(config.in_dim, config.patch_len, num_patches, config.d_model, config.head_dp if config.head_dp else 0)
        self.head.requires_grad_(True)

        self.mask = torch.linspace(0, num_patches-2, num_patches-1, dtype=torch.int64).unsqueeze(0)

    def forward(self, x):
        #mask = self.mask.repeat(x.shape[0], 1).to(x.device) 
        mask = None
        patched, h = self.encoder(x, mask=mask)

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
    
class PatchTradLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr
    
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