import torch 
import torch.nn as nn
import lightning as L
from torchmetrics.regression import MeanSquaredError
from models.base_model import PatchTrADencoder
from models.metric import StreamL2Loss

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
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

        # out: bs x nvars x target_len

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

        if config.revin:
            self.revin = RevIN(num_features=config.in_dim, eps=1e-5, affine=True)
        else:
            self.revin = None

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

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x, mode='norm')

        _, h = self.encoder(x)

        prediction = self.head(h)
        prediction = prediction.permute(0, 2, 1)
        if self.revin is not None:
            prediction = self.revin(prediction, mode='denorm')
        return prediction
    
class JePatchTST(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr
<<<<<<< HEAD
        self.criterion = nn.MSELoss() 
        self.test_mse = MeanSquaredError()
=======
        self.criterion = nn.MSELoss()

        self.l2loss = StreamL2Loss()
>>>>>>> 106d1c8f30a37b8800b23206547b33819c5ff477
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
<<<<<<< HEAD
        self.test_mse.update(pred, y)
    
    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        self.test_mse.reset()
        self.log("mse", test_mse, prog_bar=True)    
=======
        self.l2loss.update(pred, y)
    
    def on_test_epoch_end(self):
    
        l2loss = self.l2loss.compute()
        self.log("l2loss", l2loss, prog_bar=True)
        self.l2loss.reset()
>>>>>>> 106d1c8f30a37b8800b23206547b33819c5ff477
