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

import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, List, Tuple, Any
import numpy as np
from functools import partial
from datetime import datetime
from datasets import load_dataset, Audio
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

PATH = 'E:/hf'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_activation(act: str) -> nn.Module:
    """Get activation function by name."""
    act_map = {
        "gelu": nn.GELU(), 
        "relu": nn.ReLU(), 
        "sigmoid": nn.Sigmoid(), 
        "tanh": nn.Tanh(), 
        "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), 
        "softplus": nn.Softplus(), 
        "softshrink": nn.Softshrink(), 
        "leaky_relu": nn.LeakyReLU(), 
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

@dataclass
class Dimensions:
    vocab: int
    ctx: int
    dims: int
    head: int
    layer: int
    mels: int
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

def plot_waveform(x=None, w=None, p=None, per=None, sample_idx=0, sr=16000, hop_length=160, 
                                 title="", markers=None, marker_labels=None, 
                                 show_voiced_regions=True, show_energy=False):
    num_plots = sum([x is not None, w is not None, p is not None, per is not None])
    if num_plots == 0:
        raise ValueError("No data to plot. Please provide at least one input tensor.")
    t_spans = []
    
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t_spans.append(len(w_np) / sr)
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        t_spans.append(x_np.shape[0] * hop_length / sr)
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_spans.append(len(p_np) * hop_length / sr)
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_spans.append(len(per_np) * hop_length / sr)
    max_t = max(t_spans) if t_spans else 0
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
    cu_ax = 0
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        axs[cu_ax].plot(t, w_np, color="tab:blue")
        
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
            axs[cu_ax].plot(t_energy, energy, color="red", alpha=0.7, label="Energy")
            axs[cu_ax].legend(loc='upper right')
        axs[cu_ax].set_title("Waveform")
        axs[cu_ax].set_ylabel("Amplitude")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        cu_ax += 1
    
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        axs[cu_ax].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma", 
                                   extent=[0, x_np.shape[0]*hop_length/sr, 0, x_np.shape[1]])
        axs[cu_ax].set_title("Spectrogram")
        axs[cu_ax].set_ylabel("Mel Bin")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        cu_ax += 1
    
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_p = np.arange(len(p_np)) * hop_length / sr
        axs[cu_ax].plot(t_p, p_np, color="tab:green")
        axs[cu_ax].set_title("Pitch")
        axs[cu_ax].set_ylabel("Frequency (Hz)")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[cu_ax].set_ylim([0, min(1000, p_np.max() * 1.2)])
        cu_ax += 1
    
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        axs[cu_ax].plot(t_per, per_np, color="tab:red")
        axs[cu_ax].set_title("Period (Voice Activity)")
        axs[cu_ax].set_ylabel("periodocity")
        axs[cu_ax].set_xlim([0, max_t])
        axs[cu_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[cu_ax].set_ylim([-0.05, 1.05])
        axs[cu_ax].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    
    if markers is not None:
        for i, t in enumerate(markers):
            label = marker_labels[i] if marker_labels and i < len(marker_labels) else None
            for ax in axs:
                ax.axvline(x=t, color='k', linestyle='-', alpha=0.7, label=label if i == 0 else None)
        if marker_labels:
            axs[0].legend(loc='upper right', fontsize='small')
    axs[-1].set_xlabel("t (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # type: ignore
    plt.show()
    return fig

def valid(default_value, *items):
    """Get first non-None item"""
    for item in items:
        if item is not None:
            return item
    return default_value

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
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))  # type: ignore
            init.ones_(self.weight)  
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)  # type: ignore
    
def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)  # type: ignore

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
    def __init__(self, dims, head, max_ctx=1500, radii=True, debug: List[str] = [], use_pbias=False):
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

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2), requires_grad=True if use_pbias else False)
        theta = (torch.tensor(10000, device=device, dtype=dtype))
        self.theta = nn.Parameter(theta, requires_grad=True)    
        self.theta_values = []

    def mel_scale_scalar(self, freq: float) -> float:
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def mel_scale(self, freq: Tensor) -> Tensor:
        return 1127.0 * (1.0 + freq / 700.0).log()

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
                    self.dim // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000

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
            return torch.polar(radius.unsqueeze(-1), freqs)
        else:
            return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x=None, enc=None, layer=None, feature_type="audio") -> Tensor:
        f0 = enc.get("f0") if enc is not None else None 

        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 2:
            batch, ctx = x.shape
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape # type: ignore

        if f0 is not None:
            if f0.dim() == 2:
                f0 = f0.squeeze(0) 
            theta = f0 + self.theta  
        else:
            theta = self.theta 

        freqs = self.theta_freqs(theta)
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = t[:, None] * freqs
        
        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            freqs = torch.polar(radius.unsqueeze(-1), freqs)
        else:
            radius = torch.ones_like(freqs)
            freqs = torch.polar(radius, freqs)
        
        if "radius" in self.debug and self.counter == 10:
            theta_value = theta.mean()
            radius_shape = radius.shape if 'radius' in locals() else "N/A"
            radius_mean = radius.mean() if 'radius' in locals() else 0.0
            print(f"  [{layer}] [Radius] {radius_shape} {radius_mean:.2f} [Theta] {theta_value:.2f} [f0] {f0.shape if f0 is not None else None} [Freqs] {freqs.shape} {freqs.mean():.2f} [ctx] {ctx}")
            print(f"  [{layer}] [Radius] {radius}")
        # self.theta_values.append(theta.item())
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
                radii=True,
                )
        else:
            self.rope = None

    def cos_sim(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out

    def rbf_scores(self, q, k, rbf_sigma=1.0, rbf_ratio=0.0):
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
          
    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, enc = None, layer = None, feature_type="audio", need_weights=True) -> tuple:

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

            q = self.rope.apply_rotary(q, (self.rope(x=q2, enc=enc, layer=layer)))  # type: ignore
            k = self.rope.apply_rotary(k, (self.rope(x=k2, enc=enc, layer=layer)))  # type: ignore
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if self.rbf:
            qk = self.rbf_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        if self.use_pbias:
            pbias = self.rope.pitch_bias(f0 = enc.get("f0", None) if enc is not None else None)  # type: ignore
            if pbias is not None:
                qk = qk + pbias[:,:,:q2,:q2]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if mask is not None:
            # mask = mask[:q2, :q2]#torch.tril(torch.ones(q2, q2, device=q.device))
            # audio_mask = torch.ones(q2, k2 - q2, device=q.device)
            # mask = torch.cat([mask, audio_mask], dim=-1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        
        if "multihead" in self.debug and self.counter % 100 == 0:
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape} - {qk.shape}, wv shape: {wv.shape}")
        self.counter += 1        
        return self.o(wv), qk

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
    def __init__(self, dims, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
    
    def forward(self, x):
        if not self.enabled:
            return None
        return self.gate(x)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.features = features
        self.debug = debug
        self.counter = 0
        self.dropout = 0.01

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
        self.mlp_gate = mlp_gate(dims=dims, enabled=not any([tgate, mgate, cgate]))
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims)
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, enc=None, layer=None, feature_type="audio") -> Tensor:
        
        b = torch.sigmoid(self.blend)
        ax = x + self.attn(self.lna(x), xa=xa, mask=mask, enc=enc, layer=layer)[0]  
        bx = b * ax + (1 - b) * x
        cx = self.lnb(bx)
        dx = self.mlp(cx)
        ex = self.t_gate(cx) if not None else self.default(self.m_gate(cx), self.mlp_gate(cx))
        fx = x + ex + dx
        gx = self.lnc(fx)
        return gx
            
class FEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1, use_rope=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        
        act_fn = get_activation(act)
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)
        
        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(
                    dims=self.head_dim,
                    head=self.head,
                    use_2d_axial=True,
                    spec_shape=spec_shape, debug=[])
            else:
                self.rope = rotary(
                    dims=self.head_dim,
                    head=self.head,
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
        if feature_type == "spectrogram" and self.rope is not None:
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
        
        act_fn = get_activation(act)
        
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
                head=self.head,
                debug=[])
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
        
        act_fn = get_activation(act)
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=7, stride=8, padding=3), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=5, stride=4, padding=2), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=5, padding=2), act_fn)
        
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                head=self.head,
                debug=[])
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

class SpeechTransformer(nn.Module):
    _seen = set()  
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, debug: List[str], features: List[str], act: str = "gelu"):
        super(SpeechTransformer, self).__init__()

        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.features = features
        self.dropout = 0.01
        self.sequential = "sequential" in debug
        act_fn = get_activation(act)

        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.register_buffer("audio_embedding", sinusoids(ctx, dims))
        
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

        self.block = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", debug=debug, features=features)
            for _ in range(layer)])  
        
        self.blend = nn.Parameter(torch.tensor(0.5))
        self.ln_dec = RMSNorm(dims)
        
        def get_mask(text_ctx, aud_ctx):
            mask = torch.tril(torch.ones(text_ctx, text_ctx, device=device), diagonal=0)
            audio_mask = torch.ones(text_ctx, aud_ctx - text_ctx, device=device)
            full_mask = torch.cat([mask, audio_mask], dim=-1)
            return full_mask
        self.register_buffer("mask_ax", get_mask(ctx, ctx), persistent=False)

        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0)        
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, enc, layer="encoder"):
        enc = dict_to(enc, device, dtype)

        x = enc.get("input_ids").long()
        x = self.token(x) + self.positional[:x.shape[1]]
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = {}
        out.update(enc)

        for f in self.features:
            if f in enc and f in self.blocks:
                xa = enc[f]
                for block in self.blocks[f]: # type: ignore
                    xa = block(xa, enc=enc, layer=layer)
                out[f] = xa
                xa = xa + self.audio_embedding[:xa.shape[1]]

        for block in self.block:
            mask = self.mask[:x.shape[1], :x.shape[1]]
            x = block(x, xa=None, mask=mask, enc=None, layer=layer)

        for f in self.features:
            if f in enc:
                mask = self.mask_ax[:x.shape[1], :xa.shape[1]]
                for block in self.block:
                    out = block(x, xa=xa, mask=mask, enc=None, layer=layer)
                if self.sequential:
                    x = out
                else:
                    a = torch.sigmoid(self.blend)
                    x = a * out + (1 - a) * x

        x = self.ln_dec(x)   
        return x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param

        self.SpeechTransformer = SpeechTransformer(
            vocab=param.vocab,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            debug=param.debug,
            features=param.features,
            act=param.act,
            )
        
    def forward(self,
        labels=None,
        input_ids=None,
        waveform: Optional[torch.Tensor]=None,
        spectrogram: Optional[torch.Tensor]=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        envelope: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, Optional[torch.Tensor]]:

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
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids

        logits = self.SpeechTransformer(encoder_inputs)

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
            "Conv2d": 0, "SEBlock": 0, "SpeechTransformer": 0, 
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
            elif isinstance(module, SpeechTransformer):
                self.init_counts["SpeechTransformer"] += 1
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
        
        for i in range(max_length - 1):
            with torch.no_grad():
                encoder_inputs["input_ids"] = ids
                logits = self.SpeechTransformer(encoder_inputs)
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

def setup_tokenizer(token: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("./tokenizer.json")
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

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            elif isinstance(ids, np.ndarray):
                ids = ids.tolist()
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

def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
          
    return waveform

def extract_features(batch, tokenizer, sample_rate=16000, hop_length=256, **dataset_config):

    audio = batch["audio"]
    sr = audio["sampling_rate"]
    wav = load_wave(wave_data=audio, sample_rate=sr)

    dataset_config = {
        "hop_length": 256,
        "f_min": 150,
        "f_max": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sample_rate": 16000,
        "pad_mode": "constant",
        "center": True, 
        "power": 1.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False}

    transform = torchaudio.transforms.MelSpectrogram(
        **dataset_config
        )
    
    mel_spectrogram = transform(wav)      
    log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    spec = (log_mel + 4.0) / 4.0
    spec = torch.tensor(spec)
    # batch["spectrogram"] = spec
    
    wav_np = wav.numpy().astype(np.float64)  
    f0, t = pw.dio(wav_np, sample_rate, frame_period=hop_length/sample_rate*1000)
    f0 = pw.stonemask(wav_np, f0, t, sample_rate)
    f0 = torch.from_numpy(f0)  

    labels = tokenizer.encode(batch["transcription"])

    return {
        "spectrogram": spec,
        "f0": f0,
        "labels": labels,
        # "waveform": wav,
        # "pitch": f0,
    }

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, **dataset_config):

        if sanity_check:
            test = load_dataset(
                "google/fleurs", "en_us", token=token, split="test[:10]", trust_remote_code=True
            ).cast_column("audio", Audio(sample_rate=sample_rate))

            dataset = test.map(
                lambda x: extract_features(x, tokenizer, **dataset_config),
                remove_columns=test.column_names)
            dataset = dataset(remove_columns=["audio", "transcription"]).with_format(type="torch")
            train_dataset = dataset
            test_dataset = dataset
        else:

            cache_dir = "./processed_datasets"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file_train = os.path.join(cache_dir, "train.arrow")
            cache_file_test = os.path.join(cache_dir, "test.arrow")

            if os.path.exists(cache_file_train) and os.path.exists(cache_file_test):
                from datasets import Dataset
                train_dataset = Dataset.load_from_disk(cache_file_train)
                test_dataset = Dataset.load_from_disk(cache_file_test)
                return train_dataset, test_dataset   

        def filter_func(x):
            return (0 < len(x["transcription"]) < 512 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 1500 * 160)

        raw_train = load_dataset(
            "google/fleurs", "en_us", token=token, split="train[:1000]", trust_remote_code=True)
        raw_test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test[:100]", trust_remote_code=True)

        raw_train = raw_train.filter(filter_func)
        raw_test = raw_test.filter(filter_func)

        raw_train = raw_train.cast_column("audio", Audio(sampling_rate=sample_rate))
        raw_test = raw_test.cast_column("audio", Audio(sampling_rate=sample_rate))

        train_dataset = raw_train.map(
            lambda x: extract_features(x, tokenizer, **dataset_config),
            remove_columns=raw_train.column_names)
        test_dataset = raw_test.map(
            lambda x: extract_features(x, tokenizer, **dataset_config),
            remove_columns=raw_test.column_names)

        train_dataset.save_to_disk(cache_file_train)
        test_dataset.save_to_disk(cache_file_test)
        return train_dataset, test_dataset

@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', 1)
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)

        for key in all_keys:
            if key == "labels":
                labels_list = [f["labels"] for f in features]
                max_len = max(len(l) for l in labels_list)
                all_ids, all_labels = [], []
                for label in labels_list:
                    label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                    decoder_input = [bos_token_id] + label_list
                    label_eos = label_list + [eos_token_id]
                    input_len = max_len + 1 - len(decoder_input)
                    label_len = max_len + 1 - len(label_eos)
                    padded_input = decoder_input + [pad_token_id] * input_len
                    padded_labels = label_eos + [pad_token_id] * label_len
                    all_ids.append(padded_input)
                    all_labels.append(padded_labels)
                batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

            elif key in ["spectrogram", "waveform", "pitch", "f0", "envelope", "phase"]:
                
                items = [f[key] for f in features if key in f]
                items = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in items]
                max_len = max(item.shape[-1] for item in items)
                padded = []
                for item in items:
                    pad_width = max_len - item.shape[-1]
                    if pad_width > 0:
                        pad_item = F.pad(item, (0, pad_width), mode='constant', value=pad_token_id)
                    else:
                        pad_item = item
                    padded.append(pad_item)
                batch[key] = torch.stack(padded)
                if key == "spectrogram":
                    batch["spectrogram"] = batch[key]
        return batch

def levenshtein(reference_words, hypothesis_words):
    m, n = len(reference_words), len(hypothesis_words)
    dist_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        dist_matrix[i][0] = i
    for j in range(n+1):
        dist_matrix[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if reference_words[i-1] == hypothesis_words[j-1]:
                dist_matrix[i][j] = dist_matrix[i-1][j-1]
            else:
                substitution = dist_matrix[i-1][j-1] + 1
                insertion = dist_matrix[i][j-1] + 1
                deletion = dist_matrix[i-1][j] + 1
                dist_matrix[i][j] = min(substitution, insertion, deletion)
    return dist_matrix[m][n]

def wer_batch(references, hypotheses):
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.lower().split()
        errors = levenshtein(ref_words, hyp.lower().split()) 
        total_errors += errors
        total_words += len(ref_words)
    return (total_errors / total_words) * 100 if total_words > 0 else 0.0

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, optimizer=None, scheduler=None):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.tensor(pred_ids)
        pred_ids = pred_ids.argmax(dim=-1)

    pred_ids = pred_ids.tolist()
    label_ids = label_ids.tolist()
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    label_ids = [[pad_token_id if token == -100 else token for token in seq] for seq in label_ids]

    if print_pred:
        for i in range(min(num_samples, len(pred_ids))):

            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

            print(f"Pred tokens: {pred_ids[i]}")
            print(f"Label tokens: {label_ids[i]}")
            print(f"Pred: '{pred_str[i]}'")
            print(f"Label: '{label_str[i]}'")

            print("-" * 40)
            
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
    else:
        trainable_params = 0.0
        efficiency_score = 0.0

    return {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
    }

def main():
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)
    train_dataset, test_dataset = prepare_datasets(tokenizer, token)
    param = Dimensions(
        vocab=40000, ctx=2048, dims=512, head=4, layer=4,
        mels=128, act="swish", 
        debug={},
        cross_attn=True, 
        features=["spectrogram"]
        )
    
    model = Echo(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=False,
    )
    from functools import partial
    metrics_fn = partial(compute_metrics, 
    print_pred=True, 
    num_samples=1, 
    tokenizer=tokenizer, model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, eps=1e-8, weight_decay=0.025, betas=(0.9, 0.999), 
    amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_args.max_steps, eta_min=1e-9, last_epoch=-1)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset, # type: ignore
        eval_dataset=test_dataset, # type: ignore
        data_collator=DataCollator(tokenizer=tokenizer), # type: ignore 
        compute_metrics=metrics_fn,
        optimizers=(optimizer, scheduler) # type: ignore
    )
    model.init_weights()
    trainer.train()

if __name__ == "__main__":
    main()

