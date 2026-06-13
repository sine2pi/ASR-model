import os, torch, numpy as np
from torch.nn.functional import embedding, scaled_dot_product_attention as SDPA
from typing import Iterable
from torch.nn.utils.parametrizations import weight_norm
from functools import partial
from dataclasses import dataclass
# pyrefly: ignore [missing-import]
from einops.layers.torch import Rearrange
from optimizerc import FAMScheduler2, MaxFactor, MaxFactorA, MaxFactorB, MaxFactor1, MaxFactor2
from torch.utils.data import DataLoader    
from datetime import datetime
from essentials import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

THETA = 30000.0
PATH = './cache'

@dataclass
class Dimensions:
    tokens: int
    mels: int
    dims: int
    head: int
    layer: int
    act: str
    n_type: str

def l2norm(t):
    return torch.nn.functional.normalize(t, dim = -1)

def Sequential(*modules):
    return nn.Sequential(*filter(have, modules))  

def have(a):
    return a is not None  

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, aorb(b, c))

def abcord(a, b, c, d):
    return aorb(a, aborc(b, c, d))

def no_none(xa):
    return xa.apply(lambda tensor: tensor if tensor is not None else None)

def shift_right(x):
    return torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def sinusoids(ctx, dims, theta=THETA):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta), requires_grad=False)) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32, requires_grad=False))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1), requires_grad=False)
    return positional_embedding    

class LocalNorm(nn.Module):
    def __init__(n, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, mode: str = '1', threshold: float = 0.8):
        super().__init__()
        n.size = size
        n.alpha = alpha
        n.beta = beta
        n.k = k
        n.mode = mode
        n.threshold = threshold

    def forward(n, input: Tensor, confidence=None) -> Tensor:
        if input.numel() == 0:
            return input

        div = input.mul(input).unsqueeze(1) 
        
        pad_len = n.size // 2
        if n.mode == "1":
            div = F.avg_pool1d(div, kernel_size=n.size, stride=1, padding=pad_len)

        elif n.mode == "2":
            avg_d = F.avg_pool1d(div, kernel_size=n.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=n.size, stride=1, padding=pad_len)
            condition = (max_d > 2.0 * avg_d).float()
            div = (condition * max_d) + ((1 - condition) * avg_d)

        elif n.mode == "3":
            avg_d = F.avg_pool1d(div, kernel_size=n.size, stride=1, padding=pad_len)
            max_d = F.max_pool1d(div, kernel_size=n.size, stride=1, padding=pad_len)
            
            if confidence is None:
                div = avg_d
            else:
                conf_mask = (confidence > n.threshold).float().unsqueeze(1)
                div = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        div = div.narrow(2, 0, input.size(1)).squeeze(1)
        denom = div.mul(n.alpha).add(n.k).pow(n.beta)
        return input / denom

class LayerNorm(nn.Module):
    def __init__(n, dims, eps=1e-5):
        super().__init__()
        n.dims = dims
        n.eps = eps
        n.gamma = nn.Parameter(torch.ones(dims))
        n.beta = nn.Parameter(torch.zeros(dims))

    def forward(n, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (n.dims,), n.gamma, n.beta, n.eps)
        return x.transpose(1, -1)

class AdaLN(nn.Module):
    def __init__(n, dims):
        super().__init__()

        n.norm = nn.LayerNorm(dims, elementwise_affine=False)

        n.mlp = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(), 
            nn.Linear(dims, 2 * dims)
        )

        nn.init.zeros_(n.mlp[-1].weight)
        nn.init.zeros_(n.mlp[-1].bias)

    def forward(n, x, condition=None):
        if condition is None:
             return n.norm(x)

        scale_bias = n.mlp(condition)
        gamma, beta = torch.chunk(scale_bias, 2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return n.norm(x) * (1 + gamma) + beta

def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    norm_map = {
        "layernorma": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        "layernorman": lambda: LayerNorm(dims=dims),
        "linearnormie": lambda: LinearNorm(in_dim=dims, out_dim=dims, bias=False),
        "adanormal": lambda: AdaLN(dims=dims),
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=dims),
        "localnorm": lambda: LocalNorm(size=5),
        }
   
    norm_func = norm_map.get(n_type)
    if norm_func:
        return norm_func()
    else:
        return nn.LayerNorm(dims) 

def get_activation(act: str) -> nn.Module:

    act_map = {
        "gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU(),}
    return act_map.get(act, nn.GELU())

def gammatone(dims, head, min_freq=200.0, max_freq=8000.0):
    head_dim = dims // head
    f = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
    return f / 1000

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, layer, act, n_type, norm=False, enc=False):
        super().__init__()

        n.norm = get_norm(n_type, dims) if norm else nn.Identity()
        n.local_norm = get_norm("localnorm", dims) if norm else nn.Identity()        

        act_fn = get_activation(act)

        n.conv1 = nn.Sequential(
            nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1),
             n.norm
             )
        n.conv2 = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), 
            n.local_norm
            )

        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()

        n.encoder = nn.ModuleList()
        for _ in range(layer):
            n.encoder.append(
                nn.Sequential(
                    act_fn, 
                    weight_norm(nn.Conv1d(dims, dims, kernel_size=3, padding=1)),
                    n.norm,
                    act_fn,
                    nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), 
                    act_fn,
                    nn.Dropout(0.1)
                ))

    def _process_feature(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)

        for layer in n.encoder:
            x = layer(x)

        x = x.permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1]).to(device, dtype)
        x = n.norm(x)
        return n.EncoderLayer(x)
               
    def forward(n, x):
        if isinstance(x, TensorDict):
            return x.apply(n._process_feature)
        else:
            return n._process_feature(x)

class rotary(nn.Module):
    def __init__(n, dims, head):
        super().__init__()

        n.head_dim = dims // head
        n.head = head
        n.dims = dims
        n.lin = nn.Linear(dims, n.head_dim // 2, bias=True)

    def gammatone(n, min_freq=200.0, max_freq=8000.0):
        head_dim = n.dims // n.head
        freqs = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
        return freqs / 1000

    def wideband(n, max_freq=8000.0):
        head_dim = n.dims // n.head
        mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
        mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 700 * mel_scale / 1000

    def compute_f(n, x=None, mask=None):
        if mask is None:
            scale = gammatone(n.dims, n.head)
            return x.mean(dim=-1) * scale / 1000 if x is not None else 200 * scale / 1000
        else: 
            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype, requires_grad=False) / n.head_dim * torch.log(torch.tensor(x.mean(dim=-1) * THETA if x is not None else THETA, requires_grad=False))

    def forward(n, x=None, xa=None, mask=None): 
        t = torch.arange(x.shape[2], device=device, dtype=dtype).float()
        f = torch.einsum('i,j->ij', t,  n.compute_f(mask=mask))
        m = torch.norm(xa, dim=-1, keepdim=True)
        # m = n.lin(xa)
        # m = torch.sigmoid(n.lin(xa)) ** t
        # m = torch.sigmoid(n.lin(xa)) 
        if mask is None:
            f = torch.polar(m, f)
        else: 
            f = torch.polar(m, f)

        x1 = x[..., :f.shape[-1]*2]
        x2 = x[..., f.shape[-1]*2:]
        s = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * f
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(s)
        return torch.cat([x1.type_as(x), x2], dim=-1)
    
## for reference - lets give rope a memory

# # mem_state comes from the ConvNeXt/Fuser block processing the PAST chunk
# delta_m, delta_f = n.memory_proj(mem_state).chunk(2, dim=-1)

# # m is your current magnitude, f is your standard time-step angle
# warp_m = m * torch.sigmoid(delta_m)  # Modulate amplitude
# warp_f = f + delta_f                 # Shift the phase of time itself

# # The rotary embedding is now dynamically warped by the emotional memory
# f_complex = torch.polar(warp_m, warp_f) 

# class SimpleMaskDownSampler(nn.Module):
#     """
#     Progressively downsample a mask by total_stride, each time by stride.
#     Note that LayerNorm is applied per *token*, like in ViT.

#     With each downsample (by a factor stride**2), channel capacity increases by the same factor.
#     In the end, we linearly project to embed_dim channels.
#     """

#     def __init__(
#         self,
#         embed_dim=256,
#         kernel_size=4,
#         stride=4,
#         padding=0,
#         total_stride=16,
#         activation=nn.GELU,
#         # Option to interpolate the input mask first before downsampling using convs. In that case, the total_stride is assumed to be after interpolation.
#         # If set to input resolution or None, we don't interpolate. We default to None to be safe (for older configs or if not explicitly set)
#         interpol_size=None,
#     ):
#         super().__init__()
#         num_layers = int(math.log2(total_stride) // math.log2(stride))
#         assert stride**num_layers == total_stride
#         self.encoder = nn.Sequential()
#         mask_in_chans, mask_out_chans = 1, 1
#         for _ in range(num_layers):
#             mask_out_chans = mask_in_chans * (stride**2)
#             self.encoder.append(
#                 nn.Conv2d(
#                     mask_in_chans,
#                     mask_out_chans,
#                     kernel_size=kernel_size,
#                     stride=stride,
#                     padding=padding,
#                 )
#             )
#             self.encoder.append(LayerNorm2d(mask_out_chans))
#             self.encoder.append(activation())
#             mask_in_chans = mask_out_chans

#         self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
#         self.interpol_size = interpol_size
#         if self.interpol_size is not None:
#             assert isinstance(self.interpol_size, (list, tuple)), (
#                 f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
#             )
#             self.interpol_size = list(interpol_size)
#             assert len(self.interpol_size) == 2

#     def forward(self, x: torch.Tensor):
#         if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
#             x = F.interpolate(
#                 x.float(),
#                 size=self.interpol_size,
#                 align_corners=False,
#                 mode="bilinear",
#                 antialias=True,
#             )
#         return self.encoder(x)

# # Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
# class CXBlock(nn.Module):
#     r"""ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch

#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """

#     def __init__(
#         self,
#         dim,
#         kernel_size=7,
#         padding=3,
#         drop_path=0.0,
#         layer_scale_init_value=1e-6,
#         use_dwconv=True,
#     ):
#         super().__init__()
#         self.dwconv = nn.Conv2d(
#             dim,
#             dim,
#             kernel_size=kernel_size,
#             padding=padding,
#             groups=dim if use_dwconv else 1,
#         )  # depthwise conv
#         self.norm = LayerNorm2d(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(
#             dim, 4 * dim
#         )  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = (
#             nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             if layer_scale_init_value > 0
#             else None
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = self.norm(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x

# class SimpleFuser(nn.Module):
#     def __init__(self, layer, num_layers, dim=None, input_projection=False):
#         super().__init__()
#         self.proj = nn.Identity()
#         self.layers = get_clones(layer, num_layers)

#         if input_projection:
#             assert dim is not None
#             self.proj = nn.Conv2d(dim, dim, kernel_size=1)

#     def forward(self, x):
#         # normally x: (N, C, H, W)
#         x = self.proj(x)
#         for layer in self.layers:
#             x = layer(x)
#         return x

# class SimpleMaskEncoder(nn.Module):
#     def __init__(
#         self,
#         out_dim,
#         mask_downsampler,
#         fuser,
#         position_encoding,
#         in_dim=256,  # in_dim of pix_feats
#     ):
#         super().__init__()

#         self.mask_downsampler = mask_downsampler

#         self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
#         self.fuser = fuser
#         self.position_encoding = position_encoding
#         self.out_proj = nn.Identity()
#         if out_dim != in_dim:
#             self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

#     def forward(
#         self,
#         pix_feat: torch.Tensor,
#         masks: torch.Tensor,
#         skip_mask_sigmoid: bool = False,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         ## Process masks
#         # sigmoid, so that less domain shift from gt masks which are bool
#         if not skip_mask_sigmoid:
#             masks = F.sigmoid(masks)
#         masks = self.mask_downsampler(masks)

#         ## Fuse pix_feats and downsampled masks
#         # in case the visual features are on CPU, cast them to CUDA
#         pix_feat = pix_feat.to(masks.device)

#         x = self.pix_feat_proj(pix_feat)
#         x = x + masks
#         x = self.fuser(x)
#         x = self.out_proj(x)

#         pos = self.position_encoding(x).to(x.dtype)

#         return {"vision_features": x, "vision_pos_enc": [pos]}

class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "pitch_tokens"]
        n.head = head
        n.head_dim = dims // head
        n.scale = 1.0 // len(features) if features else scale

        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)

    def forward(n, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, L, n.head, n.head_dim).transpose(1,2)
        k = n.k(xa).view(B, K, n.head, n.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.head_dim)
        return bias

class attention(nn.Module):
    def __init__(n, dims, head, layer, n_type=None, modal=False): 
        super().__init__()
        n.layer = layer

        n.scale = (dims // head) ** -0.25
        n.modal = modal

        n.q   = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.c   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims))

        n.conv = nn.Conv2d(head, head, 1, bias=False) if modal else nn.Identity()
        n.ln = get_norm(n_type, dims // head)
        n.rot = rotary(dims, head)

    def forward(n, x, xa=None, mask=None, pt=None, skip=False, pattern=None, window=3, zero=False, pitch_bias=None): 
        
        b, c, d = x.shape
        k, v = n.kv(aorb(xa, x))
        q = n.q(x)

        if zero:
            if n.rbf:
                qk = n.rbf_scores(q * n.scale, k * n.scale, rbf_sigma=1.0, rbf_ratio=0.3)
            if n.use_pbias:
                pbias = pitch_bias(xa) 
                if pbias is not None:
                    qk = qk + pbias[:,:,:q,:q]

            token_ids = k[:, :, :, 0]
            zscale = torch.ones_like(token_ids)
            fzero = torch.clamp(F.softplus(n.fzero), n.minz, n.maxz)
            zscale[token_ids.float() == n.pad_token] = fzero
            
            if mask is not None:
                if mask.dim() == 4:
                    mask = mask[0, 0]
                mask = mask[:q, :k] if xa is not None else mask[:q, :q]
                qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

            qk = qk * zscale.unsqueeze(-2)
            w = F.softmax(qk, dim=-1).to(q.dtype)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if pt is not None:
            c = n.c(pt)
            b, h, c, d = q.shape 
            t = torch.zeros(b, h, c, c, device=device, requires_grad=False)

            for i in range(c):
                for j in range(c):
                    start = max(0, min(i, j) - window)
                    end = min(c, max(i, j) + window)
                    
                    for k in range(start, end): 
                        score = (q[:, :, i, :] * k[:, :, j, :] * c[:, :, k, :]).sum(dim=-1)
                        t[:, :, i, j] += score

            q = q * n.scale + t
            k = k * n.scale + t

        else:
            q = q * n.scale 
            k = k * n.scale

        q, k = n.rot(q, xa=x if pt is None else pt, mask=mask), n.rot(k, xa=xa if xa is not None else x, mask=mask)  
        a = SDPA(n.ln(q), n.ln(k), v, is_causal=have(mask))

        if n.modal and xa is not None:
            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb = n.q(x), n.q(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)
            b = SDPA(n.ln(qa), n.ln(kb), vb, is_causal=have(mask))
            c = SDPA(n.ln(qb), n.ln(ka), va, is_causal=have(mask))
            return n.out(a), n.out(n.conv(b)), n.out(n.conv(c))
        else:
            return n.out(a)

class STthreshold(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste = STthreshold.apply

class voltage_gate(nn.Module):
    def __init__(n, dims, mem=64, thresh=0.5):
        super().__init__()
        n.mkey = nn.Parameter(torch.randn(mem, dims))
        n.mval = nn.Parameter(torch.randn(mem, 1))
        n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        
        n.threshold = nn.Parameter(torch.tensor(thresh, dtype=dtype), requires_grad=False)
        n.concat = nn.Linear(2, 1, device=device, dtype=dtype)

    def forward(n, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x_val = n.concat(torch.cat((torch.matmul(key, n.mval), n.mlp(x)), dim=-1))
        
        survival_mask = apply_ste(x_val, n.threshold)
        return survival_mask, x_val

    def update_threshold(n, loss, current_loss_ema, lr=0.01):
        if loss > current_loss_ema:
            n.threshold.sub_(lr)
        else:
            n.threshold.add_(lr)
        n.threshold.data = torch.clamp(n.threshold.data, 0.05, 0.95)

class NodeOfRanvier(nn.Module):
    def __init__(n, dims, expand=2):
        super().__init__()
        n.dims = dims
        n.expand = expand
        n.parallel = nn.ModuleList([nn.Linear(dims, dims) for _ in range(expand)])
        n.network = nn.Linear(dims, dims)
        n.relu = nn.ReLU()

    def forward(n, x):
        features = torch.stack([pathway(x) for pathway in n.parallel])
        weights = torch.softmax(n.network(x), dim=-1)
        weighted =  torch.sum(weights * features.unsqueeze(2), dim=-1)
        return n.relu(weighted)

class MacroPolicyNet(nn.Module):
    def __init__(n, dims, max_jump=2):
        super().__init__()
        n.net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.SiLU(),
            nn.Linear(128, max_jump + 1)
        )
        
    def forward(n, pooled_features):
        return F.softmax(n.net(pooled_features), dim=-1)

class MyelinatedSheath(nn.Module):
    def __init__(n, dims, head, layer, mini_hc=False, hc_expansion_rate=2):
        super().__init__()
        n.layer = layer
        n.dims = dims
        n.learn_jumps = True  
        n.jump_statistics = {0: 0, 1: 0, 2: 0} 
        
        n.shared_head = AdaptiveSpan(dims, head)
        n.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        n.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        
        n.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)
        
        n.layers = nn.ModuleList()
        for i in range(layer):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'voltage_gate': voltage_gate(dims, mem=64, thresh=0.3),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
            }

            if mini_hc:
                layer_dict['ranvier'] = NodeOfRanvier(dims, expand=hc_expansion_rate)
            else:
                layer_dict['ranvier'] = None

            n.layers.append(nn.ModuleDict(layer_dict))

        n.policy_net = MacroPolicyNet(dims, max_jump=2)

        n.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4), 
            nn.SiLU(), 
            nn.Linear(dims * 4, dims)
        )

        n.mlp_ln = nn.LayerNorm(dims)
        n.dendrites = AdaptiveSpan(dims, head, max_dist=1, sharpen=True, temp_scale=0.01)

    def forward(n, x): 
        batch, ctx = x.shape[:2]
        original_x = x
        
        work_mem = n.work_mem.expand(batch, -1, -1)
        
        pooled = x.mean(dim=1)
        policy = n.policy_net(pooled)
        
        history = []
        i = 0

        while i < n.layer:
            layer = n.layers[i]
            
            ion, survival_logits = layer['voltage_gate'](x)
            mask_layer = ion.expand(-1, ctx, n.dims)
            
            px = layer['ln'](x)  

            if layer['adapter'] is not None:
                adapted_px = layer['adapter'](px)
            else:
                adapted_px = px

            if layer['ranvier'] is not None:
                layer_out = layer['ranvier'](adapted_px)
            else:
                layer_out = adapted_px
                
            gate_val = layer['gate'](px)
            x = x + gate_val * (layer_out * mask_layer)

            mem = x.mean(dim=1, keepdim=True)
            mem_val = n.mem_gate(mem)
            work_mem = mem_val * work_mem + (1 - mem_val) * mem
            
            potential = ion.mean()
            jump_grad = 1.0
            
            if potential < 0.1 and i < n.layer - 1:
                action = 1
                
            elif i < n.layer - 1:
                if n.learn_jumps:

                    jump_decisions = F.gumbel_softmax(policy, tau=1.0, hard=True)
                    action = jump_decisions.argmax(dim=-1).item()
                    jump_grad = jump_decisions[0, action] 
                else:    
                    action = torch.multinomial(policy, 1).squeeze(-1).item()
            else:
                action = 0
                
            if action in n.jump_statistics:
                n.jump_statistics[action] += batch
            else:
                n.jump_statistics[action] = batch
                
            if action > 0:
                jump_distance = action
                i_next = min(i + jump_distance + 1, n.layer)
                jump_weight = n.jump_weights[min(jump_distance-1, 2)]               
                jump_injection = jump_weight * original_x + (1-jump_weight) * work_mem.expand(-1, ctx, -1)
                x = x + (jump_injection * jump_grad) 
                
                i = i_next
                history.append({'layer': i, 'status': 'jumped_to'})
            else:
                x = x * jump_grad
                i += 1
                history.append({'layer': i, 'status': 'processed'})
        
        # x = n.dendrites(x)
        mlp_gate = n.mlp_gate(x)
        mlp_output = n.mlp(n.mlp_ln(x))
        x = x + mlp_gate * mlp_output
        jmp = {'jump_history': history}
        return x, jmp

class gate(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        n.features = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.top = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x, num=2):
        types, indices = torch.topk(n.top(x), num, dim=-1)
        type = torch.zeros_like(n.features(x))
        type.scatter_(-1, indices, torch.nn.functional.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class tgate(nn.Module):
    def __init__(n, dims, num_types=2):
        super().__init__()

        n.ga = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        n.cs = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))

    def forward(n, x):
        types = n.cs(x)
        ga = torch.stack([g(x) for g in n.ga], dim=-1)
        return  torch.sum(ga * types.unsqueeze(2), dim=-1)

class router(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()
        n.num_types = num_types
        n.top = nn.Linear(dims * num_types, num_types)
        n.soft = nn.Sequential(nn.Linear(dims * num_types, num_types), nn.Softmax(dim=-1))
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def weights(n, router_top, router_input, num=2):
        types, indices = torch.topk(router_top, num, dim=-1)
        type = torch.zeros_like(router_top)
        type.scatter_(-1, indices, F.softmax(types, dim=-1))
        soft_selection = n.soft(router_input)
        alpha = torch.sigmoid(n.alpha)
        return alpha * type + (1 - alpha) * soft_selection

    def forward(n, *modalities):
        stack = torch.stack(modalities, dim=-1)
        input = stack.view(stack.shape[0], stack.shape[1], -1)
        weights = n.weights(n.top(input), input)
        return torch.sum(stack * weights.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, num_types=3):
        super().__init__()

        n.layer = layer - 1 
        n.ln = get_norm(n_type="adalayernorm", dims=dims)

        n.act_fn = get_activation(act)
        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        
        n.attn = attention(dims, head, layer, n_type=n_type)
        n.router = router(dims, num_types=num_types)
        n.jump = MyelinatedSheath(dims, head, layer)

        n.mlp = nn.Sequential(n.ln, tgate(dims, num_types=num_types), 
                              nn.Linear(dims, dims*num_types), get_activation(act), nn.Linear(dims*num_types, dims), n.ln)

    def forward(n, x, xa=None, mask=None, pt=None):
        x, jmp  = n.jump(n.ln(x))
        x = n.router(*[x for _ in range(n.layer)]) + n.attn(n.ln(x), mask=mask, pt=pt)
        if xa is not None:
            xa = xa + n.audio(xa.shape[1], xa.shape[-1]).to(device, dtype)
            xa, jmp = n.jump(n.ln(xa))
            x = x + n.attn(n.ln(x), xa=n.router(*[xa for _ in range(n.layer)]), pt=pt)
        # print(jmp['jump_history']) 
        return x + n.mlp(x).to(device, dtype)

class processor(nn.Module):
    def __init__(n, tokens, mels, dims, head, layer, act, n_type, ctx=2048): 
        super().__init__()

        n.dims = dims
        n.ln = get_norm(n_type, dims)

        n.token = nn.Embedding(tokens, dims)
        n.pitch_tokens = nn.Embedding(1024, dims) 
        n.position = nn.Parameter(torch.ones(ctx, dims), requires_grad=True)
        n.blend = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        n.block: Iterable[residual] = nn.ModuleList(
            [residual(dims, head, layer, act, n_type) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)
       
    def quantize_pitch(n, xa = None, pt = None, num_bins = 256, v_min = -2.0, v_max = 2.0, embedding=False):
        # this isnt working
        indices = ((pt - v_min) / (v_max - v_min) * (num_bins - 1)).round()
        tensor = torch.clamp(indices, 0, num_bins - 1)
        tensor = torch.polar(xa, tensor) if xa is not None else tensor
        tensor = torch.view_as_real(tensor) if xa is not None else tensor
        return nn.Embedding(tensor, n.dims) if embedding else tensor

    def forward(n, x, xa=None, seq=False) -> Tensor:
        blend = torch.sigmoid(n.blend)
        mask = n.mask[:x.shape[1], :x.shape[1]]

        x1 = n.token(x)    

        if xa['pt'] is not None:
            pt = n.quantize_pitch(pt=xa['pt'])
            x2 = n.pitch_tokens(pt)
            x1 = x1 + x2 
        else:
            pt = None #  lets pass the dict im thinking of using the raw features for something downstream

        x = (x1 + n.position[:x.shape[-1]]).to(device, dtype)

        for i in n.block:
            a = i(x, mask=mask, pt=pt)
            b = i(a, xa=i(xa['a']), pt=pt)
            c = i(b, xa=i(xa['b']), pt=pt)
            d = i(c, xa=i(xa['c']), pt=pt)

            # for j in [(xa), (xb), (xc)]: e = i(x, xa=i(j, pt=pt))
            # e = torch.mean(torch.stack([xa, xb, xc]), dim=0)

            e = a + b + c
            f = torch.cat([d, e], dim=1)
            g = i(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:])
        
        x = g if seq else blend * (d) + (1 - blend) * g if g is not None else a
        return (n.ln(x) @ torch.transpose(n.token.weight.to(dtype), 0, 1)).float()

class Model(nn.Module):
    def __init__(n, param: Dimensions):
        super().__init__()

        n.param = param
        n.processor = processor(
            tokens=param.tokens,
            mels=param.mels,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            act=param.act,
            n_type=param.n_type,
            )

        n.enc = AudioEncoder(param.mels, param.dims, param.head, param.layer, param.act, param.n_type, norm=False, enc=False)

        n.layer = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer += 1        

    def forward(n, labels=None, text_ids=None, spectrogram=None, pitch=None, waveform=None, pitch_tokens=None):

        fx = next((t for t in (pitch, spectrogram, waveform) if t is not None), None)
        xa = TensorDict({
            'a': aborc(pitch, spectrogram, waveform),
            'b': aborc(spectrogram, pitch, waveform),
            'c': aborc(waveform, pitch, spectrogram),
            'pt': pitch_tokens,
            }, batch_size=fx.shape[0])

        x = text_ids
        xa = n.enc(no_none(xa))
        output = n.processor(x, xa, seq=False)

        loss = None
        if labels is not None: 
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": output, "loss": loss}

    @torch.no_grad()
    def generate(n, spectrogram=None, pitch=None, waveform=None, pitch_tokens=None, max_new_tokens=150):
        n.eval()
        fx = next((t for t in (pitch, spectrogram, waveform) if t is not None), None)

        xa_dict = TensorDict({
            'a': aborc(pitch, spectrogram, waveform),
            'b': aborc(spectrogram, pitch, waveform),
            'c': aborc(waveform, pitch, spectrogram),
        }, batch_size=fx.shape[0]).to(device)
        
        xa_enc = n.enc(no_none(xa_dict))
        if pitch_tokens is not None:
            xa_enc['pt'] = pitch_tokens 

        y = torch.tensor([[1]], dtype=torch.long, device=device).repeat(fx.shape[0], 1)
        
        for _ in range(max_new_tokens):
            logits = n.processor(y, xa_enc.clone(), seq=True) 
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            y = torch.cat((y, next_token), dim=1)
            if (next_token == 2).all():
                break
                
        return y

    def _init_w(n, m):
        n.counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0, "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, m in n.named_modules():
            if isinstance(m, nn.RMSNorm):
                n.counts["RMSNorm"] += 1
            if isinstance(m, nn.LayerNorm):
                n.counts["LayerNorm"] += 1                
            elif isinstance(m, nn.Linear):
                n.counts["Linear"] += 1

    def init_w(n):
        print("Initializing model w...")
        n.apply(n._init_w)
        print("Initialization summary:")
        for module_type, count in n.counts.items():
            if count > 0:
                print(f"{module_type}: {count}")
    
def main():

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    metadata_file = "H:/DEV/datasets/lb1/metadata.csv"
    data_dir = "H:/DEV/datasets/lb1"

    # metadata_file = "./datasets/LJSpeech1000/metadata.csv"
    # data_dir = "./datasets/LJSpeech1000"
    # metadata_file = "./datasets/cv17_1000/metadata.csv"
    # data_dir = "./datasets/cv17_1000"

    log_dir = os.path.join('./logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("H:/DEV/sam3_motion/sine2pi/ASR-model/tokenizer.json") 
    
    extract_args = {

        "spectrogram": False,
        "pitch": True,
        "waveform": False,
        "pitch_tokens": False,
        "harmonics": False,
        "aperiodics": False,
        "hop_length": 160,
        "sample_rate": 16000,
        "mels": 128
    }

    param = Dimensions(tokens=40000, mels=128, dims=512, head=4, layer=4, act="gelu", n_type="layernorm")

    dataset = prepare_datasets(metadata_file, data_dir, tokenizer, extract_args=extract_args)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    model = Model(param).to('cuda')

    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)
    Collator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=1, 
        collate_fn=Collator, 
        num_workers=0,
    )

    eval_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=1, 
        collate_fn=Collator, 
        num_workers=0,
    )

    main_params = []
    jump_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'jump' in name or 'policy_net' in name or 'micro_filter' in name:
            jump_params.append(p)
        else:
            main_params.append(p)

    optimizer = MaxFactor([
        {'params': main_params, 'bias': 1.0}, 
        {'params': jump_params, 'bias': 2.0}  
    ], lr=0.025, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=0.01, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=False, cap=0.0)

    scheduler = FAMScheduler2(optimizer, warmup_steps=10, total_steps=100, 
                 decay_start=None, warmup_start=1e-6, eta_min=1e-6, last_epoch=-1) 

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    train_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_dataloader,
        eval_loader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metric_fn=metrics_fn,
        max_steps=100,
        device="cuda",
        acc_steps=1,
        clear_cache=False,
        log_interval=10,
        eval_interval=10,
        save_interval=0,
        warmup_interval=10,
        checkpoint_dir=log_dir,
        log_dir=log_dir,
        generate=False,
        clip_grad_norm=0.0,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
