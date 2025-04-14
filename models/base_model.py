import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
import math
import lightning as L 
from copy import deepcopy

from models.mask import apply_masks
from models.schedulers import WarmupCosineSchedule, CosineWDSchedule
from models.vicreg import VICRegLoss2

class Patcher(nn.Module):
    def __init__(self, window_size, patch_len):
        super().__init__()
        assert window_size % patch_len == 0, "window size must be divisible by patch length"
        self.window_size = window_size
        self.patch_len = patch_len
        self.patch_num = window_size// patch_len
        self.shape = {"window_size":self.window_size,
                              "patch_len":self.patch_len,
                              "patch_num":self.patch_num}

    def forward(self, window):

        # Input: 

        # x: bs x nvars x window_size

        # Output:

        # out: bs x nvars x patch_num x patch_len 
        patch_window = rearrange(window, 'b c (pn pl) -> b c pn pl', pl=self.patch_len)
        return patch_window
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def PositionalEncoding(q_len, d_model):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe, requires_grad=False)

class _ScaledDotProduct(nn.Module):
    def __init__(self, d_model, n_heads, attn_dp=0.):
        super().__init__()

        self.attn_dp = nn.Dropout(attn_dp)
        head_dim = d_model//n_heads
        self.scale = head_dim**(-0.5)

    def forward(self, q, k, v, prev=None):
        
        # Input: 

        # q: bs x nheads x num_patches x d_k
        # k: bs x nheads x d_k x num_patches
        # v: bs x nheads x num_patches x d_v
        # prev: bs x nheads x num_patches x num_patches

        # Output:

        # out: bs x nheads x num_patches x d_v
        # attn_weights: bs x nheads x num_patches x num_patches
        # attn_scores: bs x nheads x num_patches x num_patches

        attn_scores = torch.matmul(q, k)*self.scale

        if prev is not None: attn_scores+=prev

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dp(attn_weights)

        out = torch.matmul(attn_weights, v)
        
        return out, attn_scores
    

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dp=0., proj_dp=0., qkv_bias=True):
        super().__init__()
        d_k = d_model//n_heads if d_k is None else d_k
        d_v = d_model//n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, n_heads*d_v, bias=qkv_bias)

        self.sdp = _ScaledDotProduct(d_model=d_model, n_heads=n_heads, attn_dp=attn_dp)

        self.to_out = nn.Sequential(nn.Linear(n_heads*d_v, d_model), nn.Dropout(proj_dp))

    def forward(self, Q, K=None, V=None, prev=None):

        # Input: 

        # Q: bs x num_patches x d_model
        # K: bs x num_patches x d_model
        # V: bs x num_patches x d_model
        # prev: bs x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x num_patches x num_patches

        bs = Q.size(0)
        if K is None: K = Q.clone()
        if V is None: V = Q.clone()
        
        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)

        out, attn_scores = self.sdp(q, k, v, prev=prev)

        out = out.transpose(1, 2).contiguous().view(bs, -1, self.n_heads*self.d_v)
        out = self.to_out(out)

        return out, attn_scores
    
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.self_attn = _MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, attn_dp=attn_dp, proj_dp=dp)
        self.attn_dp = nn.Dropout(attn_dp)
        self.norm_attn = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dp),
                                nn.Linear(d_ff, d_model))
        
        self.ffn_dp = nn.Dropout(dp)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev):

        # Input: 

        # src: bs x num_patches x d_model
        # prev: bs x n_heads x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x nheads x num_patches x num_patches

        src, scores = self.self_attn(Q=src, prev=prev)
        src = self.attn_dp(src)
        src = self.norm_attn(src)

        src2 = self.ff(src)

        src = src + self.ffn_dp(src2)
        src = self.norm_ffn(src)

        return src, scores
    

class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0., n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, 
                                                     d_ff=d_ff, attn_dp=attn_dp, dp=dp) for _ in range(n_layers)])
        
    def forward(self, x):

        # Input: 

        # x: bs x num_patches x d_model

        # Output:

        # out: bs x num_patches x d_model
        out=x
        prev=None
        for layer in self.layers:
            out, prev = layer(out, prev=prev)
        return out
    

class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, n_heads, n_layers=3, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        self.patch_num, self.patch_len = patch_num, patch_len

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = PositionalEncoding(q_len=patch_num, d_model=d_model)
        self.dp=nn.Dropout(dp)
        
        self.encoder = TSTEncoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff, attn_dp=attn_dp, dp=dp, n_layers=n_layers)

    def forward(self, x, mask=None):

        # Input: 

        # x: bs x nvars x num_patches  x patch_len

        # Output:

        # out: bs x nvars x d_model x num_patches

        n_vars = x.shape[1]
        x = self.W_P(x) # bs x nvars x num_patches x d_model

        x = torch.reshape(x, (x.shape[0]*x.shape [1], x.shape[2], x.shape[3])) # bs*nvars x num_patches x d_model    (channel indep)
        x = self.dp(x+self.W_pos)
        if mask is not None:
            x = apply_masks(x, mask)
        x = self.encoder(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # bs x nvars x num_patches x d_model

        return x  # bs x nvars x num_patches x d_model
    

class PatchTrADencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        window_size = config.ws
        patch_len = config.patch_len
        d_model = config.d_model
        n_heads = config.n_heads
        n_layers = config.n_layers
        d_ff = config.d_ff
        attn_dp=0.
        dp=0.3

        self.patcher = Patcher(window_size=window_size, patch_len=patch_len)
        shape = self.patcher.shape
        patch_num = shape["patch_num"]

        self.encoder = TSTiEncoder(patch_num=patch_num, patch_len=patch_len, d_model=d_model, 
                                   n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, attn_dp=attn_dp,
                                   dp=dp)
        self.tp = Transpose(1, 2)
    

    def forward(self, x, mask=None):
        # Input: 

        # x: bs x window_size x nvars
        
        patched = self._get_patch(x) # bs x nvars x patch_len x patch_num
        
        h = self.encoder(patched, mask) # bs x nvars x patch_num x d_model

        return patched, h
    
    def _get_patch(self, x):
        x = self.tp(x) # bs x nvars x window_size
        patched = self.patcher(x) # bs x nvars x patch_num x patch_len
        return patched
    

class TSTiPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        window_size = config.ws
        patch_len = config.patch_len
        patch_num = window_size//patch_len
        d_model = config.d_model
        predictor_dim = config.predictor_dim
        n_heads = config.predictor_nheads
        n_layers = config.predictor_nlayers
        d_ff = config.predictor_dff
        attn_dp=0.
        dp=0.

        self.W_proj = nn.Linear(d_model, predictor_dim)
        self.W_pos = PositionalEncoding(q_len=patch_num, d_model=predictor_dim)
        self.dp=nn.Dropout(dp)
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim), requires_grad=True)
        
        self.encoder = TSTEncoder(d_model=predictor_dim, n_heads=n_heads, d_ff=d_ff, attn_dp=attn_dp, dp=dp, n_layers=n_layers)
        self.norm = nn.LayerNorm(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, d_model)

    def forward(self, x, mask_enc, mask_dec):

        # Input: 

        # x: bs x nvars x num_patches x d_model
        # mask_enc: bs x  indices 
        # mask_dec: bs x num_patches-indices

        # Output:

        # out: bs x nvars x d_model x num_patches

        bs, n_vars, N_ctxt, d_model = x.shape
        tar_len = mask_dec.shape[1]

        x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3])) # bs*nvars x num_patches x d_model    (channel indep)
        x = self.W_proj(x)

        pe = self.W_pos.unsqueeze(0).repeat(mask_enc.size(0), 1, 1)
        mask_enc = mask_enc.unsqueeze(-1).repeat(1, 1, pe.size(-1))
        pe = pe.gather(dim=1, index=mask_enc)
        pe = pe.repeat_interleave(repeats=n_vars, dim=0)
        x = self.dp(x+pe)

        pe = self.W_pos.unsqueeze(0).repeat(mask_dec.size(0), 1, 1)
        mask_dec = mask_dec.unsqueeze(-1).repeat(1, 1, pe.size(-1))
        pe = pe.gather(dim=1, index=mask_dec)
        pe = pe.repeat_interleave(repeats=n_vars, dim=0)
        pe = pe + self.mask_token 

        x = torch.cat([x, pe], dim=1)

        x = self.encoder(x)
        x = x[:, N_ctxt:, :]

        x = self.norm(x)
        x = self.predictor_proj(x)
        x = x.reshape(bs, n_vars, tar_len, d_model) # bs x nvars x tar_len x d_model
        return x 


class LitJEPA(L.LightningModule):
    def __init__(self, config):
        super().__init__() 
        self.encoder = PatchTrADencoder(config)
        self.predictor = TSTiPredictor(config)
        self.init_weights()

        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.epochs = config.epochs

        self.automatic_optimization = False

        self.batch_size = config.batch_size

        ema = config.ema
        ipe = config.len_loader
        num_epochs = config.epochs
        ipe_scale = config.ipe_scale
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale) for i in range(int(ipe*num_epochs*ipe_scale)+1))
        self.start_lr = config.start_lr
        self.ref_lr = config.ref_lr
        self.final_lr = config.final_lr
        self.warmup = config.warmup
        self.wd = config.wd
        self.final_wd = config.final_wd
        self.ipe = ipe
        self.ipe_scale = ipe_scale

        self.save_path = config.save_path

        self.optimizer = torch.optim.AdamW(self.parameters())
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=int(self.ipe//10), 
                                         start_lr=self.start_lr, ref_lr=self.ref_lr, final_lr=self.final_lr,
                                         T_max=int(self.ipe))
        self.wd_scheduler = CosineWDSchedule(self.optimizer, ref_wd=self.wd, T_max=int(self.ipe*self.epochs*self.ipe_scale), final_wd=self.final_wd)

        self.vicreg = VICRegLoss2()
        
        self.coeff_pred, self.coeff_std, self.coeff_cov = config.coeff_pred, config.coeff_std, config.coeff_cov

    def training_step(self, batch, batch_idx):
        windows, mask_enc, mask_pred = batch
        optimizer = self.optimizers()

        lr = self.scheduler.step()
        wd_lr = self.wd_scheduler.step()

        with torch.no_grad():
            _, h = self.target_encoder(windows)
            h = F.layer_norm(h, (h.size(-1),))
            h = h.reshape(h.shape[0]*h.shape[1], h.shape[2], h.shape[3])
            h = apply_masks(h, mask_pred)

        _, z = self.encoder(windows, mask_enc)

        # Before this was computed in the prediction space
        std_loss, cov_loss = self.vicreg(z)
        self.log('Std Loss (between individus)', std_loss)
        self.log('Cov Loss (between features)', cov_loss)
        #

        z = self.predictor(z, mask_enc, mask_pred)
        z = z.reshape(z.shape[0]*z.shape[1], z.shape[2], z.shape[3])
        pred_loss = F.smooth_l1_loss(z, h)
        self.log('Pred Loss', pred_loss)    

        loss = self.coeff_pred*pred_loss + self.coeff_std*std_loss + self.coeff_cov*cov_loss
        self.log('Total Loss', loss)

        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m)*param_q.detach().data)

        self.log("lr", lr)
        self.log("wd_lr", wd_lr)

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
    
    def on_train_epoch_end(self):
        model_path = f"{self.save_path}_{self.current_epoch+1}.ckpt"
        self.save_encoder(model_path)

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved at {path}")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)