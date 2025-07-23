import os
import math
import warnings
import logging
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tensordict import TensorDict
from typing import Optional, Dict, Union, List, Tuple
import numpy as np
from functools import partial
from datetime import datetime
from tensordict import TensorDict
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from echoutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

@dataclass
class Dimensions:
    vocab: int
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str
    debug: List[str]
    features: List[str]
    tokenizer: str

class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, radii=False, debug: List[str] = [], use_pbias=False, axial=False, spec_shape=None):

        super(rotary, self).__init__()
        self.use_pbias = use_pbias
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.radii = radii
        self.debug = debug
        self.counter = 0
        self.last_theta = None
        self.axial = axial

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2), requires_grad=True if use_pbias else False)
        theta = (torch.tensor(10000, device=device, dtype=dtype))
        self.theta = nn.Parameter(theta, requires_grad=True)    
        self.theta_values = []

        if axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)

    def pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)))
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def theta_freqs(self, theta):
        if theta.dim() == 0:
            theta = theta.unsqueeze(0)
        freq = (theta.unsqueeze(-1) / 220.0) * 700 * (
            torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                    self.head_dim // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000
        return freq

    def _apply_radii(self, freqs, f0, ctx):
        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            L = radius.shape[0]
            if L != ctx:
                F = L / ctx
                idx = torch.arange(ctx, device=f0.device)
                idx = (idx * F).long().clamp(0, L - 1)
                radius = radius[idx]
                return torch.polar(radius.unsqueeze(-1), freqs), radius
            else:
                return torch.polar(radius.unsqueeze(-1), freqs), radius
        else:
            return torch.polar(torch.ones_like(freqs), freqs), None

    def check_f0(self, f0, f0t, ctx):
        if f0 is not None and f0.shape[1] == ctx:
            return f0
        elif f0t is not None and f0t.shape[1] == ctx:
            return f0t
        else:
            return None         

    def axial_freqs(self, ctx):
        if not self.axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
    
        t = torch.arange(ctx, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    def forward(self, x=None, en=None, f=None, layer=None) -> Tensor:
        ctx=x
        f0 = en.get("f0") if en is not None else None 
        f0t = en.get("f0t") if en is not None else None 

        f0 = self.check_f0(f0, f0t, ctx)
        if f0 is not None:
            theta = f0 + self.theta  
        else:
            theta = self.theta 
        freqs = self.theta_freqs(theta)
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = t[:, None] * freqs
        freqs, radius = self._apply_radii(freqs, f0, ctx)

        if self.axial and f == "spectrogram":
            freqs_2d = self.axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)

        if "radius" in self.debug and self.counter == 10:
            print(f"  [{layer}] [Radius] {radius.shape if radius is not None else None} {radius.mean() if radius is not None else None} [Theta] {theta.mean() if theta is not None else None} [f0] {f0.shape if f0 is not None else None} [Freqs] {freqs.shape} {freqs.mean():.2f} [ctx] {ctx}")
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

    rbf = False
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 1e-7, minz: float = 1e-8, maxz: float = 1e-6, debug: List[str] = [], optim_attn=False, use_pbias=False):
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
                radii=False,
                )
        else:
            self.rope = None         
    def forward(self, x: Tensor, xa = None, mask = None, en= None, layer = None, f=None) -> tuple:

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

            q = self.rope.apply_rotary(q, (self.rope(x=q2, en=en, f=f, layer=layer)))
            k = self.rope.apply_rotary(k, (self.rope(x=k2, en=en, f=f, layer=layer)))
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if self.rbf:
            qk = self.rbf_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        if self.use_pbias:
            pbias = self.rope.pitch_bias(f0 = en.get("f0", None) if en is not None else None) 
            if pbias is not None:
                qk = qk + pbias[:,:,:q2,:q2]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if mask is not None:
            if mask.dim() == 4:
                mask = mask[0, 0]
            mask = mask[:q2, :k2] if xa is not None else mask[:q2, :q2]
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if "multihead" in self.debug and self.counter % 100 == 0:
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape} - {qk.shape}, wv shape: {wv.shape}")
        self.counter += 1        
        return self.o(wv), qk

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]

class t_gate(nn.Module):
    def __init__(self, dims, num_types=4, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.gate_projections = nn.ModuleList([
            nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            for _ in range(num_types)])
        self.type_classifier = nn.Sequential(
            Linear(dims, num_types),
            nn.Softmax(dim=-1))
    def forward(self, x):
        if not self.enabled:
            return None
        type_probs = self.type_classifier(x)
        gates = torch.stack([gate(x) for gate in self.gate_projections], dim=-1)
        comb_gate = torch.sum(gates * type_probs.unsqueeze(2), dim=-1)
        return comb_gate

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.m_key = nn.Parameter(torch.randn(mem_size, dims))
            self.m_val = nn.Parameter(torch.randn(mem_size, 1))
            self.gate_proj = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
            
    def forward(self, x):
        if not self.enabled:
            return None
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

class c_gate(nn.Module):
    def __init__(self, dims, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.s_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.w_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.p_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.e_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.ph_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.integ = Linear(dims*5, dims)
        
    def forward(self, x, features):
        if not self.enabled:
            return None
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

class mlp_gate(nn.Module):
    def __init__(self, dims, head, enabled=True, one_shot=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

    def forward(self, x, xa=None, f=None):
        if not self.enabled:
            return None
        return self.gate(x)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None, one_shot=False):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.features = features
        self.debug = debug
        self.counter = 0
        self.dropout = 0.01
        self.one_shot = one_shot

        self.blend = nn.Parameter(torch.tensor(0.5)) 
        act_fn = get_activation(act)
        self.attn = MultiheadA(dims, head, rotary_emb=True, debug=debug)

        if not any([tgate, mgate, cgate]):
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        else:
            self.mlp_gate = None
        
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
        self.t_gate = t_gate(dims=dims, num_types=4*2, enabled=tgate)
        self.m_gate = m_gate(dims=dims, mem_size=mem_size, enabled=mgate)
        self.c_gate = c_gate(dims=dims, enabled=cgate)
        self.mlp_gate = mlp_gate(dims=dims, head=head, enabled=not any([tgate, mgate, cgate]), one_shot=True)
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims)
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, en=None, layer=None, f=None) -> Tensor:
 
        b = torch.sigmoid(self.blend)
        ax = x + self.attn(self.lna(x), xa=xa, mask=mask, en=en, layer=layer, f=f)[0]
        bx = b * ax + (1 - b) * x
        cx = self.lnb(bx)
        dx = self.mlp(cx)
        ex = self.t_gate(cx) if not None else self.default(self.m_gate(cx), self.mlp_gate(cx))
        fx = x + ex + dx
        gx = self.lnc(fx)
        return gx
            
class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        self.head  = head
        self.hdim  = dims // head
        self.scale = scale                      
        self.q_proj = Linear(dims, dims)
        self.k_proj = Linear(dims, dims)

    def forward(self, x: Tensor, guide: Tensor, f=None) -> Tensor | None:
        B, Q, _ = x.shape
        K       = guide.size(1)
        q = self.q_proj(x ).view(B, Q, self.head, self.hdim).transpose(1,2)
        k = self.k_proj(guide).view(B, K, self.head, self.hdim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.hdim)
        return bias

class curiosity(nn.Module):
    def __init__(self, d, h, bias=True):
        super().__init__()
        self.h  = h
        self.dh = d // h
        self.qkv = nn.Linear(d, d * 3, bias=bias)
        self.qkv_aux = nn.Linear(d, d * 3, bias=bias)
        self.o  = nn.Linear(d, d, bias=bias)
        self.g  = nn.Parameter(torch.zeros(h))

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x, xa, mask=None):
        q, k, v   = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / self.dh**0.5
        dots_aux  = (q @ ka.transpose(-2, -1)) / self.dh**0.5
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)
        h_main = p  @ v
        h_aux  = pa @ va
        g = torch.sigmoid(self.g).view(1, -1, 1, 1)
        out = self.merge(h_main * (1 - g) + h_aux * g)
        return self.o(out)

class PositionalEncoding(nn.Module):
    def __init__(self, dims, ctx):
        super(PositionalEncoding, self).__init__()
        self.dims = dims
        self.ctx = ctx
        self.pe = self.get_positional_encoding(max_ctx=ctx)

    def get_positional_encoding(self, max_ctx):
        pe = torch.zeros(max_ctx, self.dims)
        position = torch.arange(0, max_ctx, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dims, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)

    def forward(self, x):
        ctx = x.size(1)
        pe = self.pe[:, :ctx, :]
        x = x * math.sqrt(self.dims)
        x = x + pe
        return x

class FEncoder(nn.Module):
    def __init__(self, mels, dims, head, layer, kernel_size, act, stride=1, use_rope=False, spec_shape=None, debug=[]):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        self.debug = debug
        act_fn = get_activation(act)
        self.attend_pitch = False

        if self.attend_pitch:
            self.q, self.k, self.v, self.o, self.scale = qkv_init(dims, head)
            self.mlp = nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.Linear(dims, dims),
            )
        else:
            self.q, self.k, self.v, self.o, self.scale = None, None, None, None, None
            self.mlp = None

        self.encoder = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, en=None, f=None, layer="audio"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)

        return x

    def forward(self, x: Tensor, en=None, f=None, layer = None):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, en=en, f=f, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)

        if self.mlp is not None:
            x = self.mlp(x)

        if self.attend_pitch:
            xa = en["input_ids"]
            if xa is not None:
                q, k, v = create_qkv(self.q, self.k, self.v, x=xa, xa=x, head=self.head)
                out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
                out = self.o(out)
                x = x + out

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False, debug=[], spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        self.debug = debug
        act_fn = get_activation(act)
        self.target_length = None
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)
            
        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, en=None, f=None, layer="audio"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x: Tensor, en= None, f=None, layer = None):
        x = self.encoder(x).permute(0, 2, 1)
        if self.target_length and x.shape[1] != self.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.target_length).transpose(1, 2)
        if self.use_rope:
            x = self.apply_rope_to_features(x, en=en, f=f, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.ln(x)
        print(f"X: {x.shape} {f}") if "encoder" in self.debug else None
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=True, debug=[], one_shot=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dims = dims
        self.dropout = 0.01
        self.use_rope = use_rope
        self.debug = debug
        act_fn = get_activation(act)
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=7, stride=1, padding=3), act_fn,
            Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
                self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)
        
    def rope_to_feature(self, x, en=None, f="pitch", layer="PEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, en=en, f=f, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
            
    def forward(self, x: Tensor, en=None, f="pitch", layer="PEncoder"):
        raw_pitch = x.clone()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            enc_dict = en if en is not None else {}
            enc_dict = dict(enc_dict)  
            enc_dict["f0"] = raw_pitch  
            max_tscale = x.mean()*300
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
            x = self.rope_to_feature(x, en=enc_dict, f=f, layer=layer)
        else:
            max_tscale = x.mean()*300
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        print(f"X: {x.shape} {f}") if "PEncoder" in self.debug else None
        return x

class theBridge(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, 
                debug: List[str], features: List[str], act: str = "gelu"): 
        super(theBridge, self).__init__()
    
        tgate = True
        mgate = False
        cgate = False

        self.debug = debug
        self.counter = 0
        self.dropout = 0.01 
        self.features = features
        self.do_blend = "no_blend" not in self.debug
        self.sequential = "sequential" in self.debug
        self.layer = layer

        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
        self.norm = RMSNorm(dims)
        self.sinusoid_pos = lambda length, dims, max_tscale: sinusoids(length, dims, 10000)
        self.rotary = rotary(dims=dims,  head=head, debug=debug, radii=False)

        with torch.no_grad():
            self.token.weight[0].zero_()
        
        act_fn = get_activation(act)
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
    
        self.blockA = nn.ModuleDict()
        self.blockA["waveform"] = nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) 
            for _ in range(layer)] if "waveform" in features else None) 

        for feature_type in ["spectrogram", "aperiodic", "harmonic"]:
            if feature_type in features:
                self.blockA[feature_type] = nn.ModuleList(
                    [FEncoder(mels=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
                    [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) for _ in range(layer)] if feature_type in features else None)
            else:
                self.blockA[feature_type] = None

        for feature_type in ["pitch", "phase"]:
            if feature_type in features:
                self.blockA[feature_type] = nn.ModuleList(
                    [PEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act_fn)] + 
                    [Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features) for _ in range(layer)] if feature_type in features else None)
            else:
                self.blockA[feature_type] = None

        self.blockB = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act=act_fn, tgate=tgate, mgate=mgate, cgate=cgate, debug=debug, features=features)
            for _ in range(layer)])

        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0) 
        self.register_buffer("mask", mask, persistent=False)
        self.norm = RMSNorm(dims)

    def forward(self, x, xa, en, feature, sequential=False) -> Tensor:
        x = self.token(x.long()) + self.positional[:x.shape[1]]    
        for block in chain(self.blockA[feature] or []):
            xa = block(x=xa, en=en, f=feature, layer="enc")
        for block in chain(self.blockB or []):                 
            x = block(x=x, xa=None, mask=self.mask, en=en, f=feature, layer="dec")                
            xc = block(x=x, xa=xa, mask=None, en=en, f=feature, layer="cross")
            a = torch.sigmoid(self.blend)
            x = a * xc + (1 - a) * x            

        if self.counter < 1 and "encoder" in self.debug:      
                shapes = {k: v.shape for k, v in en.items()}
                print(f"Step {self.counter}: mode: {list(en.keys()) }: shapes: {shapes}")
        self.counter += 1

        x = self.norm(x)
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

        return x
   
class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        
        self.processor = theBridge(
            vocab=param.vocab,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            features=param.features,
            act=param.act,
            debug=param.debug,
            )       
        
    def forward(self,
        labels=None,
        input_ids=None,
        waveform: Optional[torch.Tensor]=None,
        spectrogram: Optional[torch.Tensor]=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        f0t: Optional[torch.Tensor]=None,
        harmonic: Optional[torch.Tensor]=None,
        aperiodic: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, Optional[torch.Tensor]]:

        en= TensorDict(batch_size=[1], device=self.device, dtype=self.dtype)

        en= {}
        if f0 is not None:
            en["f0"] = f0
        if f0t is not None:
            en["f0t"] = f0t
        if harmonic is not None:
            en["harmonic"] = harmonic
        if aperiodic is not None:
            en["aperiodic"] = aperiodic
        if phase is not None:
            en["phase"] = phase
        if pitch is not None:
            en["pitch"] = pitch
        if waveform is not None:
            en["waveform"] = waveform
        if spectrogram is not None:
            en["spectrogram"] = spectrogram

        x = input_ids
        for f, xa in en.items():

            logits = self.processor(x, xa, en, f)

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
            "Conv2d": 0, "theBridge": 0, "Echo": 0, 
            "Residual": 0, "MultiheadA": 0, 
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
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
            elif isinstance(module, PEncoder):
                self.init_counts["PEncoder"] += 1
            elif isinstance(module, FEncoder):
                self.init_counts["FEncoder"] += 1
            elif isinstance(module, WEncoder):
                self.init_counts["WEncoder"] += 1
            elif isinstance(module, theBridge):
                self.init_counts["theBridge"] += 1
            elif isinstance(module, Echo):
                self.init_counts["Echo"] += 1

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
        feature = {}
        if spectrogram is not None:
            feature["spectrogram"] = spectrogram
        if waveform is not None:
            feature["waveform"] = waveform
        if pitch is not None:
            feature["pitch"] = pitch
        if envelope is not None:
            feature["envelope"] = envelope
        if phase is not None:
            feature["phase"] = phase
        if f0 is not None:
            feature["f0"] = f0
        
        for i in range(max_length - 1):
            with torch.no_grad():
                feature["input_ids"] = ids
                logits = self.SpeechTransformer(feature)
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

def main():
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer("./")

    sanity_check = False
    streaming = False
    load_saved = False
    save_dataset = False
    cache_dir = None
    extract_args = None    

    extract_args = {
        "waveform": False,
        "spec": True,
        "f0": False,
        "f0t": False,
        "pitch": True,
        "harmonics": False,
        "aperiodics": False,
        "phase_mod": False,
        "crepe": False,        
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
    }

    param = Dimensions(
        vocab=40000,
        mels=128,
        ctx=2048,
        dims=512,
        head=4,
        layer=4,
        act="swish",
        debug={"encoder"},
        features = ["spectrogram", "pitch"],
        tokenizer=tokenizer,
        )

    train_dataset, test_dataset = prepare_datasets(tokenizer, token, sanity_check=sanity_check, sample_rate=16000, streaming=streaming,
        load_saved=load_saved, save_dataset=save_dataset, cache_dir=cache_dir, extract_args=extract_args, max_ctx=param.ctx)

    model = Echo(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    from functools import partial
    metrics_fn = partial(compute_metrics,  print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)

    if sanity_check:
        training_args = Seq2SeqTrainingArguments(
            output_dir=log_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            max_steps=10,
            eval_steps=5,
            save_steps=0,
            warmup_steps=0,
            logging_steps=1,
            logging_dir=log_dir,
            eval_strategy="steps",
            save_strategy="no",
            logging_strategy="no",
            report_to=["tensorboard"],
            push_to_hub=False,
            save_total_limit=1,
            label_names=["labels"],
            save_safetensors=False,
            eval_on_start=True,
            batch_eval_metrics=False,
            disable_tqdm=False,
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
            learning_rate=1e-7,
            weight_decay=0.01,
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=log_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            max_steps=1000,
            eval_steps=100,
            save_steps=1000,
            warmup_steps=100,
            logging_steps=10,
            logging_dir=log_dir,
            logging_strategy="steps",
            eval_strategy="steps",
            save_strategy="no",
            report_to=["tensorboard"],
            push_to_hub=False,
            save_total_limit=1,
            label_names=["labels"],
            save_safetensors=False,
            eval_on_start=True,
            batch_eval_metrics=False,
            disable_tqdm=False,
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
            learning_rate=0.00025,
            weight_decay=0.025,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-8, weight_decay=training_args.weight_decay, betas=(0.9, 0.999), 
    amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_args.max_steps, eta_min=1e-9, last_epoch=-1)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=metrics_fn,
        optimizers=(optimizer, scheduler)
    )

    model.init_weights()
    trainer.train()
if __name__ == "__main__":

    main()

