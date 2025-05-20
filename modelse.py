

import os
import warnings
import logging

import torch, torchaudio
import torchcrepe
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from itertools import chain
from einops import rearrange
import numpy as np
from typing import Optional, Dict, Union, List, Tuple
from functools import partial
import gzip
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperTokenizer
import evaluate
import transformers
from dataclasses import dataclass

from torch import nn, Tensor

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0")
dtype = torch.float32

torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

extractor = None
tokenizer = None
optimizer = None
scheduler = None
model = None

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    audio_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str
    debug: bool
    cross_attn: bool
    features: dict

def plot_waveform_and_spectrogram(x=None, w=None, p=None, per=None, sample_idx=0, sr=16000, hop_length=160, 
                                 title="Waveform and Spectrogram", markers=None, marker_labels=None, 
                                 show_voiced_regions=True, show_energy=True):

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
        fig.colorbar(im, ax=axs[current_ax])
        current_ax += 1
    
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_p = np.arange(len(p_np)) * hop_length / sr
        axs[current_ax].plot(t_p, p_np, color="tab:green")
        axs[current_ax].set_title("Pitch Contour")
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

def shift_with_zeros(input_ids: torch.Tensor, pad_token_id=0, decoder_start_token_id=0):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == 0, pad_token_id)
    return shifted_input_ids 

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], eps = 1e-8, elementwise_affine = True, device=torch.device(device="cuda:0"), dtype=torch.float32):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}
        if isinstance(dims, int):
            
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        super().__init__(normalized_shape=dims, eps=eps, elementwise_affine=elementwise_affine)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **tox))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.device, x.dtype), None if self.bias is None else self.bias.to(x.device, x.dtype))

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, variable_radius=False,
                 learned_radius=False, debug=False):
        super().__init__()
        self.debug = False
        self.interpolate_factor = 10.0
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)), requires_grad=learned_freq)
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius)
            
        self.theta = nn.Parameter(torch.tensor(float(theta)), requires_grad=False)

    def forward(self, x = None, f0=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=device).float()
        else:
            t = x.float().to(self.inv_freq.device)

        if f0 is not None:
            f0_tensor = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_tensor.ndim > 1:
                f0_tensor = f0_tensor.squeeze()
            f0_mean = f0_tensor.mean()
            f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
            perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
            min_theta, max_theta = 800.0, 10000.0
            f0_theta = min_theta + perceptual_factor * (max_theta - min_theta)
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=device) / self.dims))
        else:
            inv_freq = self.inv_freq

        freqs = torch.einsum('i,j->ij', t, inv_freq)

        freqs = freqs.float()
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)

        freqs = freqs.unsqueeze(0)
        if self.debug:
            if self._counter == 1:
                print(f'ROTA -- freqs: {freqs.shape}, x: {x},  {t.shape if x is not None else None}', freqs.shape, t.shape)
            if f0 is not None and self._counter % 100 == 0:
                print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
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
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)

class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, debug=False):
        super().__init__()

        self.count = 0
        self.debug = debug
        self.pad_token = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
        self.Zfactor = nn.Parameter(torch.tensor(0.00001))
        
        self.rope = rotary(
            dims=self.head_dim,
            max_ctx = 1500,
            theta = 10000,
            learned_freq = False,
            variable_radius = False,
            learned_radius = False,
            debug = False)

    def forward(self, x: Tensor, xa = None, mask = None, return_attn=False, f0=None):

        z = default(xa, x)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if f0 is not None:
            qf = self.rope(q.size(1), f0=f0)
            kf = self.rope(k.size(1), f0=f0)
        else:
            qf = self.rope(q.size(1))
            kf = self.rope(k.size(1))

        bat, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        q = self.rope.apply_rotary(q, qf)
        k = self.rope.apply_rotary(k, kf)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        zfactor = torch.clamp(F.softplus(self.Zfactor), min=0.00001, max=0.0001)
        zscale[token_ids.float() == self.pad_token] = zfactor.to(q.device, q.dtype)
        if mask is not None:
            mask = mask[:ctx, :ctx]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        if return_attn:
            return qk, v
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        if self.debug and self.count % 100 == 0:
            print(f"Step {self.count}: x: {x.shape}, xa: {xa.shape if xa is not None else None}, mask: {mask.shape if mask is not None else None}")
        self.count += 1
        return self.o(wv), qk.detach()
    
class MultiheadB(nn.Module):
    def __init__(self, dims: int, head: int, debug=False):
        super().__init__()

        self.count = 0
        self.debug = debug
        self.pad_token = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
        self.Zfactor = nn.Parameter(torch.tensor(0.00001))
        
        self.rope = rotary(
            dims=self.head_dim,
            max_ctx = 1500,
            theta = 10000,
            learned_freq = False,
            variable_radius = False,
            learned_radius = False,
            debug = False)

    def forward(self, x: Tensor, xa = None, mask = None, return_attn=False, f0=None):

        z = default(xa, x)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if f0 is not None:
            qf = self.rope(q.size(1), f0=f0)
            kf = self.rope(k.size(1), f0=f0)
        else:
            qf = self.rope(q.size(1))
            kf = self.rope(k.size(1))

        bat, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        q = self.rope.apply_rotary(q, qf)
        k = self.rope.apply_rotary(k, kf)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        zfactor = torch.clamp(F.softplus(self.Zfactor), min=0.00001, max=0.0001)
        zscale[token_ids.float() == self.pad_token] = zfactor.to(q.device, q.dtype)
        if mask is not None:
            mask = mask[:ctx, :ctx]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        if return_attn:
            return qk, v
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        if self.debug and self.count % 100 == 0:
            print(f"Step {self.count}: x: {x.shape}, xa: {xa.shape if xa is not None else None}, mask: {mask.shape if mask is not None else None}")
        self.count += 1
        return self.o(wv), qk.detach()

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, ctx, act, cross_attn=False, debug=False):
        super().__init__()
        self.ctx = ctx
        self._counter = 0
        self.dropout = 0.1
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.cross_attn = cross_attn
        self.debug = debug

        self.blend_xa = nn.Parameter(torch.tensor(0.5))
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head)
        self.attnb = (MultiheadB(dims, head) if cross_attn else None)
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attn else None
        self.lnc = RMSNorm(dims)

    def forward(self, x, xa=None, mask=None, f0=None):
        r = x
        x = x + self.attna(self.lna(x), mask=mask, f0=f0)[0]
        if self.attnb and xa is not None:
            mask = None
            cross_out = self.attnb(self.lnb(x), xa, mask=mask, f0=f0)[0]
            blend = torch.sigmoid(self.blend_xa)
            x = blend * x + (1 - blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r
        if self.debug and self._counter % 10 == 0:
            print(f"Step {self._counter}: Blend factor: {self.blend_xa.item():.2f}, xa: {xa is not None}, mask: {mask is not None}, x: {x.shape}")
        self._counter += 1
        return x

class PitchEncoder(nn.Module):
    def __init__(self, input_dims, dims, hop_length=160):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=7, stride=8, padding=3),
            nn.GELU(),
            Conv1d(dims//4, dims//2, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            Conv1d(dims//2, dims, kernel_size=5, stride=5, padding=2),
            nn.GELU()
        )
    def forward(self, x):
        return self.encoder(x)

class WavefeatureEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act):
        super().__init__()
        self.head_dim = dims // head
        self.dropout = 0.1
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                   "tanh": nn.Tanh(), "swish": nn.SiLU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.downsample = nn.Sequential(
            Conv1d(input_dims, dims//8, kernel_size=15, stride=8, padding=7),
            act_fn,
            Conv1d(dims//8, dims//4, kernel_size=7, stride=4, padding=3),
            act_fn,
            Conv1d(dims//4, dims, kernel_size=9, stride=5, padding=4),
            act_fn,
        )
        
        self.encoder = nn.Sequential(
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims//8),
            act_fn,
            Conv1d(dims, dims, kernel_size=1),
            act_fn)
        self.positional_embedding = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)
        
    def forward(self, x, f0=None):
        x = self.downsample(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class AudioEncoder(nn.Module):
    def __init__(self, mels, dims, head, ctx, layer, act, debug, features, cross_attn=False):
        super().__init__()
        self.debug = debug
        self.features = features
        self._counter = 0
        self.dropout = 0.1
        cross_attn=False
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                  "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), 
                  "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), 
                  "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
            
        self.spec =  nn.ModuleList(
            [featureEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(dims=dims, head=head, ctx=ctx, act=act, cross_attn=cross_attn, debug=debug) for _ in range(layer)]) if "spectrogram" in features else None
        
        self.wave = nn.ModuleList(
            [WavefeatureEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] +
            [Residual(dims=dims, head=head, ctx=ctx, act=act, cross_attn=cross_attn, debug=debug) for _ in range(layer)] if "waveform" in features else None
        )
        
        self.pitch = nn.ModuleList(
            [featureEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=nn.GELU(), stride=2)] +
            [Residual(dims=dims, head=head, ctx=ctx, act=act, cross_attn=cross_attn, debug=debug) for _ in range(layer)] if "pitch" in features else None
        )

    def forward(self, encoder_inputs):
        if self._counter < 1 and self.debug:
            x = encoder_inputs.get("spectrogram")
            w = encoder_inputs.get("waveform")
            p = encoder_inputs.get("pitch")
            plot_waveform_and_spectrogram(x=x, w=w, p=p, hop_length=128)

        if "f0" in self.features:
            f0 = encoder_inputs.get("pitch") 
        else:
            f0 = None
            
        feature_outputs = {}

        if "spectrogram" in encoder_inputs:
            x = encoder_inputs["spectrogram"]
            for block in self.spec:
                x = block(x, f0=f0)
            feature_outputs["spectrogram"] = x

        if "waveform" in encoder_inputs:
            w = encoder_inputs["waveform"]
            for block in chain(self.wave or []):
                w = block(w, f0=f0)
            feature_outputs["waveform"] = w

        if "pitch" in encoder_inputs:
            p = encoder_inputs["pitch"]
            for block in chain(self.pitch or []):
                p = block(p, f0=f0)
            feature_outputs["pitch"] = p

        if self._counter % 10 == 0 and self.debug:
            feature_names = list(feature_outputs.keys())
            feature_shapes = {k: v.shape for k, v in feature_outputs.items()}
            print(f"Step {self._counter}: Processed modalities: {feature_names}")
            print(f"Feature shapes: {feature_shapes}")
            
        self._counter += 1
        return feature_outputs

class featureEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1, name=None):
        super().__init__()
        self.head_dim = dims // head  
        self.dropout = 0.1 
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
        act_fn = act_map.get(act, nn.ReLU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2),
            act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims),
            act_fn
        )
        
        self.act = act_fn
        self.positional_embedding = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)
        self._norm = RMSNorm(dims)

    def forward(self, x, f0=None):
        x = self.encoder(x).permute(0, 2, 1)
        x = x + self.positional_embedding(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._norm(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, layer: int, dims: int, head: int, ctx: int, cross_attn, features, debug=False):
        super().__init__()
        self._counter = 0
        self.dropout = 0.1
        self.debug = debug
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        
        self.self_attn_layers = nn.ModuleList([
            Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=False, debug=debug)
            for _ in range(layer)])
        
        self.processors = nn.ModuleDict({
            
            "spectrogram": nn.ModuleList([
                Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=cross_attn, debug=debug)
                for _ in range(layer)] if "spectrogram" in features else None
        ),
            "waveform": nn.ModuleList([
                Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=cross_attn, debug=debug)
                for _ in range(layer)] if "waveform" in features else None
        ),
            "pitch": nn.ModuleList([
                Residual(dims=dims, head=head, ctx=ctx, act="gelu", cross_attn=cross_attn, debug=debug)
                for _ in range(layer)] if "pitch" in features else None
        )})
        
        self.feature_blend = nn.ParameterDict({
            "spectrogram": nn.Parameter(torch.tensor(0.5)),
            "waveform": nn.Parameter(torch.tensor(0.5)),
            "pitch": nn.Parameter(torch.tensor(0.5)),
            })
        
        self.ln_dec = RMSNorm(dims, **tox)
        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0)        
        self.register_buffer("mask", mask, persistent=False)
     
    def forward(self, x, encoder_outputs, feature_order=None) -> Tensor:
        if feature_order is None:
            feature_order = ["pitch", "spectrogram", "waveform"]
        x = x.to(device=device)
        mask = self.mask[:x.shape[1], :x.shape[1]]
        x = (self.token_embedding(x) + self.positional_embedding[:x.shape[1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for layer in self.self_attn_layers:
            x = layer(x, mask=mask)
        text_only = x
        for feature in feature_order:
            if feature in encoder_outputs:
                encoding = encoder_outputs[feature]
                output = x
                for layer in self.processors[feature]:
                    output = layer(output, 
                    xa=encoding, mask=mask)
                alpha = torch.sigmoid(self.feature_blend[feature])
                x = alpha * output + (1 - alpha) * x
                if self.debug:
                    print(f"Blend weight for {feature}: {alpha.item():.3f}")
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param

        self.shared = nn.ModuleDict({
            "rotary_encoder": rotary(dims=param.audio_dims // param.audio_head, max_ctx=param.audio_ctx),
            "rotary_decoder": rotary(dims=param.text_dims // param.text_head, max_ctx=param.text_ctx),
        })

        self.param_tracking_paths = {
            "RotationA": "encoder.blockA.0.attna.rotary.inv_freq",
            "RotationB": "decoder.rotary.inv_freq",
            "Silence": "encoder.blockA.0.attna.Factor",
        }

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.audio_idx,
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
        # f0=None,
    ) -> Dict[str, torch.Tensor]:

        decoder_input_ids = input_ids
        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch

        encoder_outputs = self.encoder(encoder_inputs)
        logits = self.decoder(input_ids, encoder_outputs)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = loss_fn(
                shifted_logits.view(-1, shifted_logits.shape[-1]), 
                shifted_labels.view(-1))
            
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_outputs} 

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
            "Residual": 0, "MultiheadA": 0, "MultiheadB": 0, 
            "MultiheadC": 0, "MultiheadD": 0, "featureEncoder": 0,
            "WavefeatureEncoder": 0, "PitchEncoder": 0
                            }

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, MultiheadA):
                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, MultiheadB):
                self.init_counts["MultiheadB"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
            elif isinstance(module, featureEncoder):
                self.init_counts["featureEncoder"] += 1
            elif isinstance(module, WavefeatureEncoder):
                self.init_counts["WavefeatureEncoder"] += 1
            elif isinstance(module, PitchEncoder):
                self.init_counts["PitchEncoder"] += 1

    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

metric = evaluate.load(path="wer")

@dataclass
class DataCollator:
    tokenizer: WhisperTokenizer
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        global extractor
        # decoder_start_token_id: int = 0
        pad_token_id = tokenizer.pad_token_id
        spec_pad = pad_token_id
        wav_pad = pad_token_id
        decoder_start_token_id = pad_token_id
        batch = {}

        if "spectrogram" in features[0] and features[0]["spectrogram"] is not None:
            spectrogram_list = [f["spectrogram"] for f in features]
            max_len_feat = max(f.shape[-1] for f in spectrogram_list)
            pad_spectrogram = []
            for feat in spectrogram_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=spec_pad)
                else:
                    pad_feat = feat
                pad_spectrogram.append(pad_feat)
            batch["spectrogram"] = torch.stack(pad_spectrogram)

        if "waveform" in features[0] and features[0]["waveform"] is not None:
            waveform_list = [f["waveform"] for f in features]
            max_len_wav = max(w.shape[-1] for w in waveform_list)
            pad_waveforms = []
            for wav in waveform_list:
                current_len = wav.shape[-1]
                padding = max_len_wav - current_len
                if padding > 0:
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    pad_wav = F.pad(wav, (0, padding), mode='constant', value=wav_pad)
                else:
                    pad_wav = wav
                pad_waveforms.append(pad_wav)
            batch["waveform"] = torch.stack(pad_waveforms)

        if "labels" in features[0] and features[0]["labels"] is not None:
            labels_list = [f["labels"] for f in features]
            max_len_labels = max(len(l) for l in labels_list)            
            all_input_ids = []
            all_labels = []

            for label in labels_list:
                label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                decoder_input = [tokenizer.pad_token_id] + label_list                
                label_with_eos = label_list + [tokenizer.pad_token_id]                
                input_padding_len = max_len_labels + 1 - len(decoder_input)
                label_padding_len = max_len_labels + 1 - len(label_with_eos)                
                padded_input = decoder_input + [tokenizer.pad_token_id] * input_padding_len
                padded_labels = label_with_eos + [tokenizer.pad_token_id] * label_padding_len                
                all_input_ids.append(padded_input)
                all_labels.append(padded_labels)            
            batch["input_ids"] = torch.tensor(all_input_ids, dtype=torch.long)
            batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

        if "pitch" in features[0] and features[0]["pitch"] is not None:
            pitch_list = [f["pitch"] for f in features]
            max_len_pitch = max(e.shape[-1] for e in pitch_list)
            pad_pitch = []
            for pitch in pitch_list:
                current_len = pitch.shape[-1]
                padding = max_len_pitch - current_len
                if padding > 0:
                    pad_pitch_item = F.pad(pitch, (0, padding), mode='constant', value=spec_pad)
                else:
                    pad_pitch_item = pitch
                pad_pitch.append(pad_pitch_item)
            batch["pitch"] = torch.stack(pad_pitch)
        return batch

def match_length(tensor, target_len):
    if tensor.shape[-1] != target_len:
        return F.interpolate(tensor.unsqueeze(0), size=target_len, mode='linear', align_corners=False).squeeze(0)
    return tensor

def ctx_to_samples(audio_ctx, hop_length):
    samples_token = hop_length * 2
    n_samples = audio_ctx * samples_token
    return n_samples

def exact_div(x, y):
    assert x % y == 0
    return x // y

def extract_features(batch, tokenizer, spectrogram=True, waveforms=True, pitch=True, f0=True, energy_contour=False, periodocity=False,
                     hop_length=128, fmin=0, fmax=8000, n_mels=128, n_fft=1024, sampling_rate=16000, pad_value=0.0,
                     pad_mode="constant", center=True, power=2.0, window_fn=torch.hann_window, mel_scale="htk", 
                     norm=None, normalized=False, debug=False):
    
    global model, extractor

    dtype = torch.float32
    device = torch.device("cuda:0")
    audio = batch["audio"]
    sampling_rate = audio["sampling_rate"]
        
    wav = torch.tensor(audio["array"]).float()
    sr = audio["sampling_rate"]
    
    if sr != sampling_rate:
        original_length = wav.shape[-1]
        target_length = int(original_length * (sampling_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = resampler(wav)
        
        if abs(wav.shape[-1] - target_length) > 1:
            new_waveform = torch.zeros((wav.shape[0], target_length), dtype=dtype, device=device)
            copy_length = min(wav.shape[1], target_length)
            new_waveform[:, :copy_length] = wav[:, :copy_length]
            wav = new_waveform

    if spectrogram:
        transform = torchaudio.transforms.MelSpectrogram(
            f_max=fmax,
            f_min=fmin,
            n_mels=n_mels,
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            norm='slaney',
            normalized=False,
            power=2.0,
            center=True, 
            mel_scale="htk",
            window_fn=torch.hann_window,  
            )
    
        mel_spectrogram = transform(wav)      
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spec = (log_mel + 4.0) / 4.0
        spec = torch.tensor(spec)

    wav = wav.unsqueeze(0)

    if pitch:
        pit = torchcrepe.predict(
            wav, 
            sampling_rate, 
            hop_length,
            fmin=150,
            fmax=600,
            model="tiny",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False, 
            device=device, 
            pad=False
        )
        
    if waveforms:
        batch["waveform"] = wav
    if pitch:
        batch["pitch"] = pit
    if spectrogram:
        batch["spectrogram"] = spec
    batch["labels"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch


def compute_metrics(eval_pred, compute_result: bool = True, print_pred: bool = False, num_samples: int = 0, tokenizer=None):
    global extractor, model, optimizer, scheduler
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0

    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if hasattr(pred_logits, "cpu"):
        pred_logits = pred_logits.cpu()
    if hasattr(label_ids, "cpu"):
        label_ids = label_ids.cpu()
    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits

    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.tensor(pred_ids)
        pred_ids = pred_ids.argmax(dim=-1)
        pred_ids = pred_ids[:, 1:].tolist()
    elif hasattr(pred_ids, "tolist"):
        if isinstance(pred_ids, torch.Tensor) and pred_ids.ndim >= 2:
            pred_ids = pred_ids[:, 1:].tolist()
        else:
            pred_ids = pred_ids.tolist()

    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()
    label_ids = [[tokenizer.pad_token_id if token == -100 else token for token in seq] for seq in label_ids]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

    if print_pred:

        for i in range(num_samples):
            print(f"Prediction: {pred_str[i]}")
            print(f"Label: {label_str[i]}")
            print("--------------------------------")
            print("")
            print(f"pred tokens  : {pred_ids[i]}")
            print(f"label tokens : {label_ids[i]}")
            print("")
            print("--------------------------------")

    all_true = []
    all_pred = []

    for pred_seq, label_seq in zip(pred_ids, label_ids):
        valid_indices = [i for i, token in enumerate(label_seq) if token != tokenizer.pad_token_id]
        
        valid_labels = [label_seq[i] for i in valid_indices if i < len(label_seq)]
        valid_preds = [pred_seq[i] for i in valid_indices if i < len(pred_seq) and i < len(valid_indices)]
        
        min_len = min(len(valid_labels), len(valid_preds))
        all_true.extend(valid_labels[:min_len])
        all_pred.extend(valid_preds[:min_len])

    acc = accuracy_score(all_true, all_pred) if all_true else 0
    pre = precision_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    rec = recall_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0) if all_true else 0
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    metrics = {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }
    return metrics

logger = logging.getLogger(__name__)

def create_model(param: Dimensions) -> Echo:
    model = Echo(param).to('cuda')
    model.init_weights()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model

def setup_tokenizer(token: str, local_tokenizer_path: str = "./tokenizer") -> WhisperTokenizer:

    tokenizer = WhisperTokenizer.from_pretrained(
        local_tokenizer_path,
        pad_token="0",
        bos_token="0",
        eos_token="0", 
        unk_token="0",
        token=token,
        local_files_only=True,
    )
    
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0
    
    return tokenizer

def prepare_datasets(
    tokenizer: WhisperTokenizer,
    token: str,
    sanity_check: bool = False,
    dataset_config: Optional[Dict] = None
) -> Tuple[any, any]:

    if dataset_config is None:
        dataset_config = {
            "spectrogram": True,
            "waveforms": True,
            "pitch": True,
            "f0": False if sanity_check else True,
            "hop_length": 128,
            "fmin": 50,
            "fmax": 2000,
            "n_mels": 128,
            "n_fft": 1024,
            "sampling_rate": 16000,
        }
    
    dataset = load_dataset(
        "google/fleurs", 
        "en_us", 
        token=token, 
        trust_remote_code=True,
        streaming=False
    )
    
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    
    if sanity_check:
        dataset = dataset["test"].take(100)
        dataset = dataset.select_columns(["audio", "transcription"])
        logger.info(f"Sanity dataset size: {dataset.num_rows}")
        
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        
        dataset = dataset.map(
            function=prepare_fn, 
            remove_columns=["audio", "transcription"]
        ).with_format(type="torch")
        train_dataset = dataset
        test_dataset = dataset
    else:
        def filter_func(x):
            return (0 < len(x["transcription"]) < 512 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 1500 * 160)
        
        dataset = dataset.filter(filter_func).shuffle(seed=4)
        logger.info(f"Dataset size: {dataset['train'].num_rows}, {dataset['test'].num_rows}")
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        train_dataset = dataset["train"]#.take(1000)
        test_dataset = dataset["test"].take(200)
        columns_to_remove = list(next(iter(dataset.values())).features)
        train_dataset = train_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
        
        test_dataset = test_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
    
    return train_dataset, test_dataset

def get_training_args(
    log_dir: str,
    batch_eval_metrics: bool = False,
    max_steps: int = 10,
    save_steps: int = 10001,
    eval_steps: int = 100,
    warmup_steps: int = 0,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
) -> Seq2SeqTrainingArguments:

    return Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=1,
        logging_steps=100,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=batch_eval_metrics,
    )

def main():
    
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)

    def sanity(sanity: bool):

        if sanity:
            training_args = get_training_args(
                log_dir,
                batch_eval_metrics=False,
                max_steps=2,
                save_steps=0,
                eval_steps=100,
                warmup_steps=0,
                learning_rate=1e-4,
                weight_decay=0.01)
        else:
            training_args = get_training_args(
                log_dir,
                batch_eval_metrics=False,
                max_steps=10000,
                save_steps=10000,
                eval_steps=2000,
                warmup_steps=1000,
                learning_rate=2.5e-4,
                weight_decay=0.01)

        return training_args
        
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        audio_dims=512,
        audio_idx=4,
        vocab=len(tokenizer),
        text_ctx=256,
        text_head=4,
        text_dims=512,
        text_idx=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=0,
        act="gelu",
        debug=False,
        cross_attn=False,
        features = {
        "spectrogram", # uncomment to use spectrogram
        "waveform", # uncomment to use waveform
        "pitch", # uncomment to use pitch
        #"f0", # uncomment to use frequency in rotary
        },
        )
    
    sanity_check = False
    training_args = sanity(sanity_check)

    dataset_config = {
        "spectrogram": True,
        "waveforms": True,
        "pitch": True,
        "hop_length": 128,
        "fmin": 150,
        "fmax": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sampling_rate": 16000,
        "pad_value": 0.0,
        "pad_mode": "constant",
        "center": True, 
        "power": 2.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False,
        "debug": True,
        }
    
    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer)
    print(f"{'Sanity check' if sanity_check else 'Training'} mode")
    train_dataset, test_dataset = prepare_datasets(
        tokenizer=tokenizer,
        token=token,
        sanity_check=sanity_check,
        dataset_config=dataset_config)
    model = create_model(param)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=metrics_fn)
    
    trainer.train()

if __name__ == "__main__":
    main()


