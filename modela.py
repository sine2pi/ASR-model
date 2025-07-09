import os
import pyworld as pw
import math
import warnings
import logging
import torch
import torchaudio
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, Tensor
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Optional, Dict, Union, List, Tuple
import transformers
from dataclasses import dataclass
from opimizer import MaxFactor
from transformers.generation.configuration_utils import GenerationConfig  
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logging.basicConfig(level=logging.ERROR)

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_idx: int
    mels: int
    aud_ctx: int
    aud_dims: int
    aud_head: int
    aud_idx: int
    act: str
    debug: List[str]
    cross_attn: bool
    features: List[str]

def get_generation_config(param):
    return GenerationConfig(
        max_length=param.text_ctx,
        pad_token_id=getattr(param, "pad_token_id", 0),
        bos_token_id=getattr(param, "bos_token_id", 1),
        eos_token_id=getattr(param, "eos_token_id", 2),
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        temperature=1.0,
        decoder_start_token_id=1,
        is_multilingual=False,
        use_cache=False,
        return_timestamps=False)

def dict_to(d, device, dtype=dtype):
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}
    
def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class RMSNorm(nn.Module):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(RMSNorm, self).__init__()
        if isinstance(dims, int):
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            init.ones_(self.weight)  
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
    
def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32 if torch.cuda.is_available() else torch.float64

def tox():
    return {"device": get_device(), "dtype": get_dtype()}

def sinusoids(length, channels, max_tscale=10000):
    assert channels % 2 == 0
    log_tscale_increment = np.log(max_tscale) / (channels // 2 - 1)
    inv_tscales = torch.exp(-log_tscale_increment * torch.arange(channels // 2))
    scaled_t = torch.arange(length)[:, np.newaxis] * inv_tscales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_t), torch.cos(scaled_t)], dim=1)


class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, theta=10000, radii=True, debug: List[str] = [], use_pbias=False):
        super(rotary, self).__init__()

        self.use_pbias = use_pbias
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.radii = radii
        self.dim = self.head_dim
        self.debug = debug
        self.counter = 0
        self.last_theta = None

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        self.theta = nn.Parameter(torch.tensor(theta, device=device, dtype=dtype), requires_grad=True)

    # def theta_freqs(self, theta):
    #     freq = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
    #     freqs = nn.Parameter(torch.tensor(freq, device=device, dtype=dtype), requires_grad=True)        
    #     return freqs

    # def mel_geodesic_rotary(f0, theta):
    #     mel_f0 = 1127.0 * torch.log(1.0 + f0 / 700.0)
    #     fisher_info = torch.var(mel_f0) + 1e-8
    #     adaptive_theta = theta * torch.sqrt(fisher_info)
    #     freqs = self.theta_freqs(adaptive_theta)
    #     return freqs

    # def compute_pitch_fisher_info(f0, window_size=10):
    #     if f0.dim() == 1:
    #         f0 = f0.unsqueeze(0)
    #     mel_f0 = 1127.0 * torch.log(1.0 + f0 / 700.0)
    #     fisher_info = torch.nn.functional.avg_pool1d(
    #         mel_f0.unsqueeze(0), 
    #         kernel_size=window_size, 
    #         stride=1, 
    #         padding=window_size//2
    #     ).squeeze(0)
    #     fisher_info = (fisher_info - fisher_info.min()) / (fisher_info.max() - fisher_info.min() + 1e-8)
    #     return fisher_info

    # def compute_advanced_fisher_info(f0, window_size=10):
    #     mel_f0 = 1127.0 * torch.log(1.0 + f0 / 700.0)
    #     local_mean = torch.nn.functional.avg_pool1d(
    #         mel_f0.unsqueeze(0), window_size, 1, window_size//2
    #     ).squeeze(0)
        
    #     local_var = torch.nn.functional.avg_pool1d(
    #         (mel_f0 - local_mean).pow(2).unsqueeze(0), 
    #         window_size, 1, window_size//2
    #     ).squeeze(0)
   
    #     fisher_info = 1.0 / (local_var + 1e-8)
    #     return fisher_info

    # def test_fisher_info(self, f0):
    #     """Test Fisher information computation."""    #     fisher_info = self.compute_pitch_fisher_info(f0)
        
    #     print(f"f0 range: {f0.min():.1f} - {f0.max():.1f}")
    #     print(f"Fisher info range: {fisher_info.min():.3f} - {fisher_info.max():.3f}")
    #     print(f"Fisher info mean: {fisher_info.mean():.3f}")
        
    #     # Visualize: high Fisher info = meaningful pitch changes
    #     return fisher_info

    # def forward(self, x=None, enc=None, layer=None, feature_type="audio"):
        
    #     if f0 is not None:
    #         # Compute Fisher information
    #         fisher_info = self.compute_pitch_fisher_info(f0)
            
    #         # Use Fisher info to weight pitch influence
    #         f0_weighted = f0 * fisher_info
            
    #         # Apply to both theta and radius
    #         f0_mean = f0_weighted.mean()
    #         theta = f0_mean + self.theta
            
    #         if self.radii:
    #             radius = f0_weighted.to(device, dtype)


    
    def theta_freqs(self, theta):
        freq = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        freqs = nn.Parameter(torch.tensor(freq, device=device, dtype=dtype), requires_grad=True)        
        return freqs

    def mel_scale_scalar(freq: float) -> float:
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def mel_scale(freq: Tensor) -> Tensor:
        return 1127.0 * (1.0 + freq / 700.0).log()

    def return_f0(self, f0=None):
        if f0 is not None:
            self.f0 = f0
            self.update_base(f0)
            return f0.squeeze(0).to(device, dtype)
        elif hasattr(self, 'f0') and self.f0 is not None:
            return self.f0.squeeze(0).to(device, dtype)
        return None

    def pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)))
        return f0_sim.unsqueeze(0).unsqueeze(0)


    def forward(self, x=None, enc=None, layer=None, feature_type="audio") -> Tensor:
        f0 = enc.get("f0") if enc is not None else None 
        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 2:
            batch, ctx = x.shape
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
        t = torch.arange(ctx, device=device, dtype=dtype)

        if f0 is not None and f0.dim() == 2:
            if f0.shape[0] == 1: 
                f0 = f0.squeeze(0)  
            else:
                f0 = f0.view(-1)        

        if f0 is not None:
            f0_mean = f0.mean()
            theta = f0_mean + self.theta
        else:
            theta = self.theta 

        freqs = self.theta_freqs(theta)

        freqs = t[:, None] * freqs[None, :]

        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            L = radius.shape[0]
            if L != ctx:
                F = L / ctx
                idx = torch.arange(ctx, device=f0.device)
                idx = (idx * F).long().clamp(0, L - 1)
                radius = radius[idx]
            freqs = torch.polar(radius.unsqueeze(-1).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)

        if "radius" in self.debug and self.counter % 100 == 0:
            theta_value = theta.item() if isinstance(theta, torch.Tensor) else theta
            print(f"  [{layer}] [Radius] {radius.shape} {radius.mean():.2f} [Theta] {theta_value:.2f} [f0] {f0.shape if f0 is not None else None} [Freqs] {freqs.shape} {freqs.mean():.2f} [ctx] {ctx}")
        
        if "theta" in self.debug and self.counter % 100 == 0:
            if self.last_theta is None or abs(self.last_theta - theta.item()) > 1.0:
                self.last_theta = theta.item()
                print(f"[Theta] {self.last_theta:.2f}")

        self.counter += 1
        return freqs.unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        if x1.ndim == 2:
            x1 = x1.unsqueeze(0)
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)


class MultiheadA(nn.Module):
    _seen = set()  
    rbf = False
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 1e-4, minz: float = 1e-6, maxz: float = 1e-3, debug: List[str] = [], optim_attn=False, use_pbias=False):
        super(MultiheadA, self).__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.use_pbias = use_pbias

        self.q = nn.Linear(dims, dims).to(device, dtype)
        self.k = nn.Linear(dims, dims, bias=False).to(device, dtype)
        self.v = nn.Linear(dims, dims).to(device, dtype)
        self.o = nn.Linear(dims, dims).to(device, dtype)

        self.pad_token = 0
        self.rotary_emb = rotary_emb
        self.minz = minz
        self.maxz = maxz
        self.zero_val = zero_val
        self.optim_attn = optim_attn        
        self.fzero = nn.Parameter(torch.tensor(zero_val, device=device, dtype=dtype), requires_grad=False)
        
        if rotary_emb:
            self.rope = rotary(
                dims=dims,
                head=head,
                debug=debug,
                radii=True,
                )
        else:
            self.rope = None
          
    def forward(self, x: Tensor, xa: Tensor = None, mask: Tensor = None, enc = None, layer = None, feature_type="audio", need_weights=True) -> tuple:

        x = x.to(device, dtype)
        if xa is not None:
            xa = xa.to(device, dtype)
        scale = (self.dims // self.head) ** -0.25
        
        z = default(xa, x).to(device, dtype)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if self.rotary_emb:   
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            q2 = q.shape[2]
            k2 = k.shape[2]

            q = self.rope.apply_rotary(q, (self.rope(x=q2, enc=enc, layer=layer)))
            k = self.rope.apply_rotary(k, (self.rope(x=k2, enc=enc, layer=layer)))
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if mask is not None:
            mask = mask[:q2, :q2]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        
        if "multihead" in self.debug and self.counter % 100 == 0:
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape} - {qk.shape}, wv shape: {wv.shape}")
        self.counter += 1        
        return self.o(wv), qk

class t_gate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gate_projections = nn.ModuleList([
            nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            for _ in range(num_types)])
        self.type_classifier = nn.Sequential(
            Linear(dims, num_types),
            nn.Softmax(dim=-1))
    def forward(self, x):
        type_probs = self.type_classifier(x)
        gates = torch.stack([gate(x) for gate in self.gate_projections], dim=-1)
        comb_gate = torch.sum(gates * type_probs.unsqueeze(2), dim=-1)
        return comb_gate

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64):
        super().__init__()
        self.m_key = nn.Parameter(torch.randn(mem_size, dims))
        self.m_val = nn.Parameter(torch.randn(mem_size, 1))
        self.gate_proj = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
        
    def forward(self, x):
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

class c_gate(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.w_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.p_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.e_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.ph_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.integ = Linear(dims*5, dims)
        
    def forward(self, x, features):
        s_feat = features.get("spectrogram", x)
        w_feat = features.get("waveform", x)
        p_feat = features.get("pitch", x)
        e_feat = features.get("envelope", x)
        ph_feat = features.get("phase", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        e = self.e_gate(x) * e_feat
        ph = self.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return self.integ(comb)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, cross_attn=True, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.cross_attn = cross_attn
        self.features = features
        self.debug = debug
        self.counter = 0
        self.dropout = 0.01
       
        self.t_gate = tgate
        self.m_gate = mgate
        self.c_gate = cgate
        self.do_blend = "no_blend" not in self.debug
        self.blend = nn.Parameter(torch.tensor(0.5)) 
        self.skip_gates = True if "skip_gates" in self.debug else False
            
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                  "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), 
                  "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), 
                  "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head, rotary_emb=True, debug=debug)
        self.attnb = (MultiheadA(dims, head, rotary_emb=True, debug=debug) if cross_attn else None)
        
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
        self.t_gate = t_gate(dims=dims, num_types=4) if tgate else None
        self.m_gate = m_gate(dims=dims, mem_size=mem_size) if mgate else None
        self.c_gate = c_gate(dims=dims) if cgate else None
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attn else None
        self.lnc = RMSNorm(dims)

        if not any([t_gate, m_gate, c_gate]):
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

    def forward(self, x, xa=None, mask=None, enc=None, layer=None, feature_type="audio") -> Tensor:

        x = x + self.attna(self.lna(x), xa=None, mask=mask, enc=enc, layer=layer)[0]
        xb = x
        if self.attnb and xa is not None:
            x = x + self.attnb(self.lnb(x), xa=xa, mask=None, enc=enc, layer=layer)[0]
            
            if self.do_blend:
                b = torch.sigmoid(self.blend)
                x = b * xb + (1 - b) * x
        
        if self.skip_gates:
            x = x + self.mlp(self.lnc(x))
        else:
            normx = self.lnc(x)
            mlp_out = self.mlp(normx)

            if self.t_gate:
                gate = self.t_gate(normx)
                x = x + gate * mlp_out
                
            elif self.m_gate:
                gate = self.m_gate(normx)
                x = x + gate * mlp_out
            
            elif self.c_gate:
                gate_output = self.c_gate(normx, self.features)
                x = x + gate_output

            else:
                if hasattr(self, 'mlp_gate'):
                    mlp_gate = self.mlp_gate(normx)
                    x = x + mlp_gate * mlp_out
                else:
                    x = x + mlp_out
                   
        return x

class FEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1, use_rope=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)
        
        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(
                    dims=self.head_dim,
                    use_2d_axial=True,
                    spec_shape=spec_shape, debug=[])
            else:
                self.rope = rotary(
                    dims=self.head_dim,
                    use_2d_axial=False, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
            
        self.norm = RMSNorm(dims)
        self._norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer=None, feature_type="audio"):
        if feature_type in ["envelope", "phase"]:
            feature_type = "spectrogram"
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        if feature_type == "spectrogram" and hasattr(self.rope, 'use_2d_axial') and self.rope.use_2d_axial:
            rope_freqs = self.rope(ctx, layer=layer, input_type="spectrogram")
        else:
            rope_freqs = self.rope(ctx, layer=layer, input_type="audio")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x

    def forward(self, x, enc=None, layer=None, feature_type="audio"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, layer=layer, feature_type=feature_type)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._norm(x)
        return x

class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.downsample = nn.Sequential(
            Conv1d(input_dims, dims//8, kernel_size=15, stride=8, padding=7), act_fn,
            Conv1d(dims//8, dims//4, kernel_size=7, stride=4, padding=3), act_fn,
            Conv1d(dims//4, dims, kernel_size=9, stride=5, padding=4), act_fn)
        
        self.encoder = nn.Sequential(
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims//8),  act_fn,
            Conv1d(dims, dims, kernel_size=1), act_fn)
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                use_2d_axial=False,
                theta=50.0, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer=None):
        if not self.use_rope or self.rope is None:
            return x
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(ctx, layer=layer, input_type="waveform")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, enc=None, layer=None, feature_type="waveform"):
        x = self.downsample(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, layer=layer)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=7, stride=8, padding=3), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=5, stride=4, padding=2), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=5, padding=2), act_fn)
        
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                use_2d_axial=False,
                theta=100.0, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer=None):
        if not self.use_rope or self.rope is None:
            return x
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(ctx, layer=layer, input_type="pitch")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, enc=None, layer=None, feature_type="pitch"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, layer=layer)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class AudioEncoder(nn.Module):
    _seen = set()  
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer: int, debug: List[str], features: List[str], act: str = "gelu"):
        super(AudioEncoder, self).__init__()

        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.features = features
        self.dropout = 0.01

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(),"tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
            
        self.blocks = nn.ModuleDict({

            "spectrogram": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] 
            if "spectrogram" in features else None), 

            "waveform": nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] 
            if "waveform" in features else None),

            "pitch": nn.ModuleList(
            [FEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] 
            if "pitch" in features else None),

            "envelope": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] 
            if "envelope" in features else None),

            "phase": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] 
            if "phase" in features else None),
            })

    def forward(self, enc, layer="encoder"):
        enc = dict_to(enc, device, dtype)
        out = {}
        out.update(enc)

        for f in self.features:
            if f in enc and f in self.blocks:
                x = enc[f]
                for block in self.blocks[f]:
                    x = block(x, enc=enc, layer=layer)
                out[f] = x

        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer: int, cross_attn: bool, 
                debug: List[str], features: List[str]): 
        super(TextDecoder, self).__init__()

        self.ctx = ctx     
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.dropout = 0.01
        self.features = features
        self.do_blend = "no_blend" not in self.debug
        self.sequential = "sequential" in self.debug

        self.token = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token.weight[0].zero_()
        self.positional = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        
        self.block = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", cross_attn=cross_attn, debug=debug, features=features)
            for _ in range(layer)])
        
        self.blocks = nn.ModuleDict({
        f: nn.ModuleList([Residual(ctx=ctx, dims=dims, head=head, act="gelu", cross_attn=cross_attn, debug=debug, features=features)
            for _ in range(layer)]) for f in features})
        
        self.blend = nn.ParameterDict({f: nn.Parameter(torch.tensor(0.5)) for f in features})
        self.ln_dec = RMSNorm(dims)
        
        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0)        
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, enc, order=None, layer='decoder') -> Tensor:

        if order is None:
            order = self.features
        
        mask = self.mask[:x.shape[1], :x.shape[1]]
        x = self.token(x) + self.positional[:x.shape[1]]
        x = F.dropout(x, p=self.dropout, training=self.training)
        

        for block in self.block:
            x = block(x, xa=None, mask=mask, enc=None, layer=layer)

        for f in order:
            if f in enc:
                xa = enc[f]
                for block in self.blocks[f]:
                    out = block(x=x, xa=xa, mask=None, enc=None, layer=layer)

                if self.sequential:
                    x = out
                else:
                    a = torch.sigmoid(self.blend[f])
                    x = a * out + (1 - a) * x
                        

        x = self.ln_dec(x)   
        return x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.aud_ctx,
            dims=param.aud_dims,
            head=param.aud_head,
            layer=param.aud_idx,
            act=param.act,
            debug=param.debug,
            features=param.features,
            )
        
        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.text_idx,
            cross_attn=param.cross_attn,
            debug=param.debug,
            features=param.features,
            )       
        
    def forward(self,
        labels=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
        spectrogram: torch.Tensor=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        envelope: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, torch.Tensor]:

        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch
        if envelope is not None:
            encoder_inputs["envelope"] = envelope
        if phase is not None:
            encoder_inputs["phase"] = phase
        if f0 is not None:
            encoder_inputs["f0"] = f0

        encoder_outputs = self.encoder(encoder_inputs)
        logits = self.decoder(input_ids, encoder_outputs)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
                
        return {"logits": logits, "loss": loss} 

    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, 
            "Residual": 0, "MultiheadA": 0, "MultiheadB - Cross Attention": 0, 
            "MultiheadC": 0, "MultiheadD": 0, "FEncoder": 0,
            "WEncoder": 0, "PEncoder": 0}

        for name, module in self.named_modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, MultiheadA):

                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
    
    def init_weights(self):
        print("Initializing model weights...")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

    def generate(self, input_ids=None, spectrogram=None, waveform=None, pitch=None, f0=None, 
        envelope=None, phase=None, tokenizer=None, max_length=128, min_length=1, device=None, **kwargs):
        if device is None:
            device = self.device
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        bos_token_id = getattr(tokenizer, "bos_token_id", 1)
        eos_token_id = getattr(tokenizer, "eos_token_id", 2)
        batch_size = 1
        for x in [spectrogram, waveform, pitch, f0, envelope, phase]:
            if x is not None:
                batch_size = x.shape[0]
                break
        ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch
        if envelope is not None:
            encoder_inputs["envelope"] = envelope
        if phase is not None:
            encoder_inputs["phase"] = phase
        if f0 is not None:
            encoder_inputs["f0"] = f0
        encoder_outputs = self.encoder(encoder_inputs)
        for i in range(max_length - 1):
            with torch.no_grad():
                logits = self.decoder(ids, encoder_outputs)
            next_token_logits = logits[:, -1, :]
            if i < min_length:
                next_token_logits[:, eos_token_id] = 0
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tokens], dim=1)
            if (next_tokens == eos_token_id).all() and i >= min_length:
                break
        return ids

    @property
    def config(self):
        class Config:
            pad_token_id = getattr(self.param, "pad_token_id", 0)
            bos_token_id = getattr(self.param, "bos_token_id", 1)
            eos_token_id = getattr(self.param, "eos_token_id", 2)
            def to_json_string(self):
                import json
                return json.dumps({
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                })
        return Config()

token = ""

param = Dimensions(
    mels=128,
    aud_ctx=1500,
    aud_head=4,
    aud_dims=512,
    aud_idx=4,
    vocab=40000,
    text_ctx=512,   
    text_head=4,
    text_dims=512,
    text_idx=4,
    act="swish",
    debug={},
    cross_attn=True,
    features = ["spectrogram"],
    )

def setup_tokenizer(token, local_tokenizer_path: str = "./"):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{local_tokenizer_path}/tokenizer.json")
    orig_encode = tokenizer.encode
    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids

    def bdec(ids_list, skip_special_tokens=True):
        results = []
        for ids in ids_list:
            if skip_special_tokens:
                if ids and ids[0] == 1:
                    ids = ids[1:]
                while ids and ids[-1] in [0, 2]:
                    ids = ids[:-1]
            results.append(tokenizer.decode(ids))
        return results

    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")
    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

class SpeechDataProcessor:
    def __init__(self, hf_dataset, tokenizer, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=256):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        self.samples = self._process_dataset(hf_dataset)
        
    def _process_dataset(self, hf_dataset):
        samples = []
        for item in hf_dataset:
            waveform = torch.tensor(item["audio"]["array"]).float()
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)
            
            spec = self.mel(waveform)
            
            wav_np = waveform.numpy().astype(np.float64)
            f0, t = pw.dio(wav_np, self.sample_rate, frame_period=self.hop_length/self.sample_rate*1000)
            f0 = pw.stonemask(wav_np, f0, t, self.sample_rate)
            f0 = torch.from_numpy(f0).float()
            
            transcription = item.get("sentence", item.get("transcription", ""))
            
            samples.append({
                "spectrogram": spec,
                "f0": f0,
                "transcription": transcription
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def collate_fn(self, batch):
        specs = [item["spectrogram"] for item in batch]
        f0s = [item["f0"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        
        max_spec_len = max(s.shape[-1] for s in specs)
        max_f0_len = max(f0.shape[-1] for f0 in f0s)
        
        padded_specs = torch.stack([
            torch.nn.functional.pad(s, (0, max_spec_len - s.shape[-1])) for s in specs
        ])
        
        padded_f0s = torch.stack([
            torch.nn.functional.pad(f0, (0, max_f0_len - f0.shape[-1])) for f0 in f0s
        ])
        
        input_ids = [self.tokenizer.encode(t) for t in transcriptions]
        max_len = max(len(ids) for ids in input_ids)
        input_ids = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            "spectrogram": padded_specs,
            "f0": padded_f0s,
            "input_ids": input_ids,
            "labels": labels,
            "transcription": transcriptions
        }
    
    def create_dataloaders(self, batch_size=1, train_split=0.8, shuffle_train=True):
        train_size = int(train_split * len(self))
        test_size = len(self) - train_size
        
        train_set, test_set = random_split(self, [train_size, test_size])
        
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=shuffle_train, 
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        return train_loader, test_loader

class SpeechTrainer:
    def __init__(self, model, data_processor, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.device = device
        self.step = 0
        self.train_losses = []
        self.eval_metrics = []
        
    def setup_optimizer(self, lr=0.00025, weight_decay=0.025, max_steps=100000):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            eps=1e-8, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.999))
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=0.0)
        self.max_steps = max_steps
        
    def train(self, max_steps=None, lr=0.00025, weight_decay=0.025, 
              batch_size=1, print_every=100, train_split=0.8):
        if max_steps is None:
            max_steps = self.max_steps if hasattr(self, 'max_steps') else 100000
            
        if not hasattr(self, 'optimizer'):
            self.setup_optimizer(lr, weight_decay, max_steps)
            
        train_loader, test_loader = self.data_processor.create_dataloaders(
            batch_size=batch_size, train_split=train_split
        )
        
        self.model.train()
        self.step = 0
        
        print(f"Starting training for {max_steps} steps...")
        
        while self.step < max_steps:
            for batch in train_loader:
                if self.step >= max_steps:
                    break
                    
                x = batch["spectrogram"].to(self.device)
                f0 = batch["f0"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                out = self.model(input_ids=input_ids, spectrogram=x, f0=f0, labels=labels)
                loss = out["loss"]
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.train_losses.append(loss.item())
                if self.step % print_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_loss = np.mean(self.train_losses[-print_every:])
                    print(f"Step {self.step}: Train loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
                self.step += 1
        print("Training completed!")
        
    def evaluate(self, max_samples=None, batch_size=1):
        train_loader, test_loader = self.data_processor.create_dataloaders(
            batch_size=batch_size, train_split=0.8)
        self.model.eval()
        total_wer = 0
        n = 0
        
        print("Starting evaluation...")
        with torch.no_grad():
            for batch in test_loader:
                if max_samples and n >= max_samples:
                    break
                    
                x = batch["spectrogram"].to(self.device)
                f0 = batch["f0"].to(self.device)
                
                pred_ids = self.model.generate(
                    spectrogram=x, f0=f0, tokenizer=self.tokenizer, max_length=32
                )
                pred_text = self.tokenizer.batch_decode(pred_ids.tolist())
                ref_text = batch["transcription"]
                w = self._calculate_wer(ref_text[0], pred_text[0])
                
                print(f"REF: {ref_text[0]}")
                print(f"PRED: {pred_text[0]}")
                print(f"WER: {w:.2f}")
                
                total_wer += w
                n += 1
                
        avg_wer = total_wer / n if n > 0 else 0.0
        print(f"\nAverage WER: {avg_wer:.2f}")
        
        self.eval_metrics.append(avg_wer)
        return avg_wer
    
    def _calculate_wer(self, ref, hyp):
        r = ref.split()
        h = hyp.split()
        d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
        
        for i in range(len(r)+1):
            d[i][0] = i
        for j in range(len(h)+1):
            d[0][j] = j
            
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
                    
        wer_value = d[len(r)][len(h)] / float(len(r)) if len(r) > 0 else 0.0
        return min(wer_value, 1.0)
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.eval_metrics = checkpoint['eval_metrics']
        print(f"Checkpoint loaded from {path}")

raw_dataset = load_dataset(
    "google/fleurs",
    "en_us",
    token=token,
    split="train",#[:2000]",
    trust_remote_code=True,
)

raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000)).shuffle()
tokenizer = setup_tokenizer(token)
data_processor = SpeechDataProcessor(raw_dataset, tokenizer)
trainer = SpeechTrainer(Echo(param), data_processor, tokenizer)
trainer.train(max_steps=1000, lr=0.00025, print_every=100)
avg_wer = trainer.evaluate(max_samples=10)
print(f"Final Average WER: {avg_wer:.2f}")
trainer.save_checkpoint("checkpoint.pt")

