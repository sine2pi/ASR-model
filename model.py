
import pyworld as pw
import os
import math, random
import warnings
import logging
import gzip
import base64
import torch
import torchaudio
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, Tensor
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Any
from functools import partial
from datetime import datetime
from datasets import load_dataset, Audio, concatenate_datasets
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import transformers
import evaluate
from dataclasses import dataclass
import matplotlib.pyplot as plt

device = torch.device(device="cuda:0")
dtype = torch.float32

extractor = None
tokenizer = None
optimizer = None
scheduler = None
model = None
Residual = None
MultiheadA = None

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
    f0_rotary: bool

def plot_waveform(x=None, w=None, p=None, per=None, sample_idx=0, sr=16000, hop_length=160, 
                                 title="", markers=None, marker_labels=None, 
                                 show_voiced_regions=True, show_energy=False):
    num_plots = sum([x is not None, w is not None, p is not None, per is not None])
    if num_plots == 0:
        raise ValueError("No data to plot. Please provide at least one input tensor.")
    time_spans = []
    
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        time_spans.append(len(w_np) / sr)
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        time_spans.append(x_np.shape[0] * hop_length / sr)
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        time_spans.append(len(p_np) * hop_length / sr)
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        time_spans.append(len(per_np) * hop_length / sr)
    max_time = max(time_spans) if time_spans else 0
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 4*num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]
    if show_voiced_regions and per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        threshold = 0.5
        for ax in axs:
            for i in range(len(per_np)-1):
                if per_np[i] > threshold:
                    ax.axvspan(t_per[i], t_per[i+1], color='lightblue', alpha=0.2, zorder=0)
    current_ax = 0
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        axs[current_ax].plot(t, w_np, color="tab:blue")
        
        if show_energy:
            frame_length = hop_length
            hop_length_energy = hop_length // 2
            energy = []
            for i in range(0, len(w_np)-frame_length, hop_length_energy):
                frame = w_np[i:i+frame_length]
                energy.append(np.sqrt(np.mean(frame**2)))
            energy = np.array(energy)
            energy = energy / np.max(energy) * 0.8 * max(abs(w_np.min()), abs(w_np.max()))  
            t_energy = np.arange(len(energy)) * hop_length_energy / sr
            axs[current_ax].plot(t_energy, energy, color="red", alpha=0.7, label="Energy")
            axs[current_ax].legend(loc='upper right')
        axs[current_ax].set_title("Waveform")
        axs[current_ax].set_ylabel("Amplitude")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        current_ax += 1
    
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        im = axs[current_ax].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma", 
                                   extent=[0, x_np.shape[0]*hop_length/sr, 0, x_np.shape[1]])
        axs[current_ax].set_title("Spectrogram")
        axs[current_ax].set_ylabel("Mel Bin")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        # fig.colorbar(im, ax=axs[current_ax])
        current_ax += 1
    
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_p = np.arange(len(p_np)) * hop_length / sr
        axs[current_ax].plot(t_p, p_np, color="tab:green")
        axs[current_ax].set_title("Pitch")
        axs[current_ax].set_ylabel("Frequency (Hz)")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[current_ax].set_ylim([0, min(1000, p_np.max() * 1.2)])
        current_ax += 1
    
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        axs[current_ax].plot(t_per, per_np, color="tab:red")
        axs[current_ax].set_title("Period (Voice Activity)")
        axs[current_ax].set_ylabel("periodocity")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[current_ax].set_ylim([-0.05, 1.05])
        axs[current_ax].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    
    if markers is not None:
        for i, t in enumerate(markers):
            label = marker_labels[i] if marker_labels and i < len(marker_labels) else None
            for ax in axs:
                ax.axvline(x=t, color='k', linestyle='-', alpha=0.7, label=label if i == 0 else None)
        if marker_labels:
            axs[0].legend(loc='upper right', fontsize='small')
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig

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

def get_tox():
    return {"device": get_device(), "dtype": get_dtype()}

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class rotary(nn.Module):

    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, radii=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = [], use_pbias = False):
        super().__init__()

        self.use_pbias = use_pbias 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32 
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.radii = radii
        f0_factor = 0.5
        self.learned_adaptation: bool = False
        pitch_scale = 1.0
        radius = 1
        
        if self.learned_adaptation:
            self.f0_scale = nn.Parameter(torch.tensor(f0_factor, device=self.device, dtype=self.dtype), requires_grad=True)
        else:
            self.register_buffer('f0_scale', torch.tensor(f0_factor))

        self.theta = nn.Parameter(torch.tensor(theta, device=self.device, dtype=self.dtype), requires_grad=True)
        self.pitch_scale = nn.Parameter(torch.tensor(pitch_scale, device=self.device, dtype=self.dtype), requires_grad=True)
        freqs = 1. / (theta ** (torch.arange(0, dims, 2, device=self.device, dtype=self.dtype)[:(dims // 2)].float() / dims))
        self.freqs = nn.Parameter(torch.tensor(freqs, device=self.device, dtype=self.dtype), requires_grad=True)
        self.radius = nn.Parameter(torch.ones(radius, device=self.device, dtype=self.dtype), requires_grad=True)

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def align_f0(self, f0, ctx):
        b, l = f0.shape
        if l == ctx:
            return f0.squeeze(0).float()  
        frames_per_token = l / ctx
        idx = torch.arange(ctx, device=self.device, dtype=torch.float32)
        src_idx = (idx * frames_per_token).long().clamp(0, l-1)
        batch_idx = torch.arange(b, device=self.device, dtype=torch.float32).unsqueeze(1)
        f0 = f0[batch_idx, src_idx]
        return f0.squeeze(0).float()
   
    def forward(self, x: Tensor, xa: Tensor = None, f0: Tensor = None, mask: Tensor = None, layer = None) -> Tensor:
        
        if isinstance(x, int):
            ctx = x
        else:
            batch, ctx, dims = x.shape
        t = torch.arange(ctx, device=self.device).float()
        
        if self.learned_adaptation:
            freqs = self.get_f0_adapted_freqs(ctx, f0)
            x_complex = torch.view_as_complex(
                x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
            x_rotated = x_complex * freqs.unsqueeze(0).unsqueeze(0)
            freqs = torch.view_as_real(x_rotated).flatten(3).type_as(x)

        if f0 is not None:
            f0_mean=f0.mean()+1e-8
            theta=f0_mean*self.pitch_scale
            freqs = 1. / (theta ** (torch.arange(0, self.dims, 2, device=self.device, dtype=self.dtype)[:(self.dims // 2)].float() /self.dims))
        else:        
            freqs = self.freqs
        freqs = torch.einsum('i,j->ij', t, freqs)
        freqs = freqs.float()

        if self.radii:
            radius = self.align_f0(f0, ctx)
            radius = torch.clamp(radius, min=40.0, max=600.0)
            radius = radius / 600.0 
            radius = radius.float()
        else:
            radius = self.radius
        freqs = torch.polar(radius.unsqueeze(-1), freqs)

        if "rotary" in self.debug and self._counter % 100 == 50:
            if f0 is not None:
                print(f"{layer}: {theta:.2f} : {f0_mean:.2f} : {ctx} ")
                if self.radii:  
                    print(f"radius: {radius} Hz, enc: {layer} Hz, ctx: {ctx}")
            self._counter += 1
        return freqs      

    @staticmethod
    def apply_rotary(x, freqs):
        multihead_format = len(freqs.shape) == 4
        if multihead_format:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)
        else:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            
            if x.ndim == 2:  
                x1 = x1.unsqueeze(0)
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                x1 = x1.squeeze(0)  
                return torch.cat([x1.type_as(x), x2], dim=-1)
            else:  
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                return torch.cat([x1.type_as(x), x2], dim=-1)

class MultiheadA(nn.Module):
    _seen = set()  
    rbf = False
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 0.0001, minz: float = 0.0, maxz: float = 0.001, debug: List[str] = [], optim_attn=False):
        
        super(MultiheadA, self).__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.debug = debug
        self._counter = 0

        self.pad_token = 0
        self.rotary_emb = rotary_emb
        self.minz = minz
        self.maxz = maxz
        self.zero_val = zero_val
        self.optim_attn = optim_attn        
        self.fzero = nn.Parameter(torch.tensor(zero_val, dtype=torch.float32), requires_grad=False)
        
        if rotary_emb:
            self.rope = rotary(
                dims=self.head_dim,
                debug = debug,
                radii=False,
                learned_pitch=False,
                learned_freq=False,
                learned_theta=False,
                learned_radius=False,
                )
        else:
            self.rope = None               
        
    def enhanced_attention_scores(self, q, k, rbf_sigma=1.0, rbf_ratio=0.0):
        scale = (self.dims // self.head) ** -0.25
        dot_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if rbf_ratio <= 0.0:
            return dot_scores
        q_norm = q.pow(2).sum(dim=-1, keepdim=True)
        k_norm = k.pow(2).sum(dim=-1, keepdim=True)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
        rbf_scores = torch.exp(-dist_sq / (2 * rbf_sigma**2))
        return (1 - rbf_ratio) * dot_scores + rbf_ratio * rbf_scores
          
    def forward(self, x: Tensor, xa: Tensor = None, f0: Tensor = None, mask: Tensor = None, layer = None) -> tuple:
        
        batch, ctx, dims = x.shape
        scale = (self.dims // self.head) ** -0.25
        
        z = xa if xa is not None else x
        q = self.q(x).to(x.dtype)
        k = self.k(z).to(x.dtype)
        v = self.v(z).to(x.dtype)

        if self.rotary_emb:   
            qf = self.rope(q.size(1), f0=f0, layer=layer)
            kf = self.rope(k.size(1), f0=f0, layer=layer)

            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            
            q = self.rope.apply_rotary(q, qf)
            k = self.rope.apply_rotary(k, kf)
            
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            batch, head, ctx, head_dim = q.shape
        
        if self.rbf:
            qk = self.enhanced_attention_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if f0 is not None and self.rope.use_pbias:
            pbias = self.rope.use_pbias(f0)
            if pbias is not None:
                qk = qk + pbias[:,:,:q.shape[2],:q.shape[2]]
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
        
        if mask is not None:
            mask = mask[:q.shape[2], :q.shape[2]]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)

        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        
        if "multihead" in self.debug and self._counter % 100 == 0:
            print(f"Step {self._counter}: Using rotary embeddings: {self.rotary_emb}")
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape}")
            print(f"Attention shape: {qk.shape}, wv shape: {wv.shape}")
        self._counter += 1        
        return self.o(wv), qk.detach()

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
        self.integ = Linear(dims*3, dims)
        
    def forward(self, x, features):
        s_feat = features.get("spectrogram", x)
        w_feat = features.get("waveform", x)
        p_feat = features.get("pitch", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        comb = torch.cat([s, w, p], dim=-1)
        return self.integ(comb)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, cross_attn=True, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.cross_attn = cross_attn
        self.features = features
        self.debug = debug
        self._counter = 0
        self.dropout = 0.01
       
        self.t_gate = tgate
        self.m_gate = mgate
        self.c_gate = cgate
        
        self.blend = nn.Parameter(torch.tensor(0.5))
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                  "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), 
                  "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), 
                  "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head, rotary_emb=True, debug=debug)
        self.attnb = (MultiheadA(dims, head, rotary_emb=True, debug=debug) if cross_attn else None)
        
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
        self.t_gate = t_gate(dims=dims, num_types=4) if t_gate else None
        self.m_gate = m_gate(dims=dims, mem_size=mem_size) if m_gate else None
        self.c_gate = c_gate(dims=dims) if c_gate else None
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attn else None
        self.lnc = RMSNorm(dims)

        if not any([t_gate, m_gate, c_gate]):
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

    def forward(self, x, xa=None, mask=None, f0=None, mode=None, layer=None):
        bln = self.blend
        x = x + self.attna(self.lna(x), xa=None, mask=mask, f0=f0, layer=layer)[0]
        
        if self.attnb and xa is not None:
            c = self.attnb(self.lnb(x), xa=xa, f0=f0, mask=None, layer=layer)[0]
            b = torch.sigmoid(bln)
            x = b * x + (1 - b) * c
        
        normx = self.lnc(x)
        mlp_out = self.mlp(normx)
        
        if self.t_gate:
            gate = self.t_gate(normx)
            x = x + gate * mlp_out
               
        elif self.m_gate:
            gate = self.m_gate(normx)
            x = x + gate * mlp_out
        
        elif self.c_gate and mode is not None:
            gate_output = self.c_gate(normx, self.features)
            x = x + gate_output

        else:
            if hasattr(self, 'mlp_gate'):
                mlp_gate = self.mlp_gate(normx)
                x = x + mlp_gate * mlp_out
            else:
                x = x + mlp_out
                
        if "residual" in self.debug and self._counter % 100 == 0:
            print(f"Step {self._counter}: Residual block output shape: {x.shape}, xa shape: {xa.shape if xa is not None else None}")        
            if self.t_gate:
                print(f"Step {self._counter}: Using t_gate: {self.t_gate}")
            elif self.m_gate:
                print(f"Step {self._counter}: Using m_gate: {self.m_gate}")
            elif self.c_gate:
                print(f"Step {self._counter}: Using c_gate: {self.c_gate}")
            else:
                print(f"Step {self._counter}: Using MLP gate: {self.mlp_gate if hasattr(self, 'mlp_gate') else None}")
        self._counter += 1      
                  
        return x

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act):
        super().__init__()
        
        self.head_dim = dims // head
        self.dropout = 0.01
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=7, stride=8, padding=3), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=5, stride=4, padding=2), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=5, padding=2),act_fn)
        
    def forward(self, x, f0=None, layer=None):
        x = self.encoder(x).permute(0, 2, 1)
        x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x
        
class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act):
        super().__init__()
        
        self.head_dim = dims // head
        self.dropout = 0.01
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.downsample = nn.Sequential(
            Conv1d(input_dims, dims//8, kernel_size=15, stride=8, padding=7), act_fn,
            Conv1d(dims//8, dims//4, kernel_size=7, stride=4, padding=3), act_fn,
            Conv1d(dims//4, dims, kernel_size=9, stride=5, padding=4), act_fn)
        
        self.encoder = nn.Sequential(
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims//8),  act_fn,
            Conv1d(dims, dims, kernel_size=1), act_fn)
        
        self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)
        
    def forward(self, x, f0=None, layer=None):
        x = self.downsample(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class FEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1):
        super().__init__()
        
        self.head_dim = dims // head  
        self.dropout = 0.01 
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)
        
        self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)
        self._norm = RMSNorm(dims)

    def forward(self, x, f0=None, layer=None):
        x = self.encoder(x).permute(0, 2, 1)
        x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._norm(x)
        return x
          
class AudioEncoder(nn.Module):
    _seen = set()  
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer: int, debug: List[str], features: List[str], 
                 f0_rotary: bool = False, act: str = "gelu"):
        super(AudioEncoder, self).__init__()

        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.debug = debug
        self._counter = 0

        self.features = features
        self.dropout = 0.01
        self.f0_rotary = f0_rotary

        self.rope = rotary(
            dims=self.head_dim,
           )

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
            
        self.blocks = nn.ModuleDict({
            "spectrogram": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "spectrogram" in features else None
            ), 
            "waveform": nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "waveform" in features else None
            ),
            "pitch": nn.ModuleList(
            [FEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "pitch" in features else None
            ),       
            "spec_envelope": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug) for _ in range(layer)] if "spec_envelope" in features else None
            ),
            "spec_phase": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug) for _ in range(layer)] if "spec_phase" in features else None),
            })

        self.f0 = nn.ModuleList([
            FEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)
            for _ in range(layer)])

    def forward(self, x, f0=None, layer="ENC"):
        if self._counter < 1:
            s = x.get("spectrogram")
            w = x.get("waveform")
            p = f0 if f0 is not None else x.get("pitch")
            plot_waveform(x=s, w=w, p=p, hop_length=128)

        enc = {}

        for y in self.features:
            if y in x and y in self.blocks:
                f = x[y]
                for block in self.blocks[y]:
                    f = block(f, f0=f0, layer=layer)
                enc[y] = f
                        
        if "encoder" in self.debug and self._counter % 100 == 0:
            names = list(x.keys())
            shapes = {k: v.shape for k, v in x.items()}
            print(f"Step {self._counter}: mode: {names}")
            print(f"shapes: {shapes}")
        self._counter += 1
        return enc

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer: int, cross_attn: bool, 
                debug: List[str], features: List[str], f0_rotary: bool = False, sequential=False): 
        super(TextDecoder, self).__init__()
                
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.debug = debug
        self._counter = 0

        self.dropout = 0.01
        self.sequential = sequential
        self.features = features
        self.f0_rotary = f0_rotary
        
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

    def forward(self, x, enc, f0=None, order=None, layer='DEC') -> Tensor:
        bln = self.blend
        x = x.to(device)
        if order is None:
            order = self.features
        mask = self.mask[:x.shape[1], :x.shape[1]]
        x = self.token(x) + self.positional[:x.shape[1]]
        x = F.dropout(x, p=self.dropout, training=self.training)
        for block in self.block:
            x = block(x, xa=None, f0=f0, mask=mask, layer=layer)
        for f in order:
            if f in enc:
                xa = enc[f]
                for block in self.blocks[f]:
                    out = block(x=x, xa=xa, f0=f0, mask=None, layer=layer)
                a = torch.sigmoid(bln[f])
                x = a * out + (1 - a) * x
        x = self.ln_dec(x)

        if "decoder" in self.debug and self._counter % 100 == 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"DECODER LAYER {name}: grad_norm={param.median():.4f}")
        self._counter += 1     
        return x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        self.count = 0

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.aud_ctx,
            dims=param.aud_dims,
            head=param.aud_head,
            layer=param.aud_idx,
            act=param.act,
            debug=param.debug,
            features=param.features,
            f0_rotary=param.f0_rotary,
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
            f0_rotary=param.f0_rotary,
            )

        all_head = torch.zeros(self.param.text_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.text_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.text_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, spectrogram: torch.Tensor):
        return self.encoder(spectrogram)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)
        
    def forward(self,
        decoder_input_ids=None,
        labels=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
        spectrogram: torch.Tensor=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        f0d: Optional[torch.Tensor]=None,
        envelope: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, torch.Tensor]:

        decoder_input_ids = input_ids
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
        # if f0 is not None:
        #     encoder_inputs["f0"] = f0

        encoder_outputs = self.encoder(encoder_inputs, f0=f0)
        logits = self.decoder(input_ids, encoder_outputs, f0=f0d)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
                
        self.count += 1
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_outputs,
            } 

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

    def register_gradient_hooks(self):
        """Add this method to your Echo model class"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    param.register_hook(lambda grad, n=name: self._print_encoder_grad(n, grad))
                elif "decoder" in name:
                    param.register_hook(lambda grad, n=name: self._print_decoder_grad(n, grad))
        
        print("Gradient debugging hooks registered")
        return self

    def _print_encoder_grad(self, name, grad):
        if grad is not None and self.count == 10:  
            norm = grad.median().item()
            print(f"ENCODER GRAD: {name} = {norm:.6f}")
        
        return None

    def _print_decoder_grad(self, name, grad):
        if grad is not None and self.count == 10: 
            norm = grad.median().item()
            print(f"DECODER GRAD: {name} = {norm:.6f}")
        return None

    def reset_counter(self):
        """Reset the internal counter for debugging purposes."""
        self._counter = 0
        print("Counter reset to 0.")

