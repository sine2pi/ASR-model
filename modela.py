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
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str
    debug: List[str]
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
    plt.tight_layout(rect=[0, 0, 1, 0.97])
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



class Sinusoids(nn.Module):
    def __init__(self, length, channels, max_tscale=10000):
        super().__init__()
        assert channels % 2 == 0
        log_tscale_increment = np.log(max_tscale) / (channels // 2 - 1)
        inv_tscales = torch.exp(-log_tscale_increment * torch.arange(channels // 2))
        scaled_t = torch.arange(length)[:, None] * inv_tscales[None, :]
        pos1 = torch.sin(scaled_t)
        pos2 = torch.cos(scaled_t)
        positions = torch.cat([pos1, pos2], dim=1) 
        self.embedding = nn.Embedding.from_pretrained(positions, freeze=False)
    def forward(self, positions):
        return self.embedding(positions)

def sinusoids(length, channels, max_tscale=10000):
    assert channels % 2 == 0
    log_tscale_increment = np.log(max_tscale) / (channels // 2 - 1)
    inv_tscales = torch.exp(-log_tscale_increment * torch.arange(channels // 2))
    scaled_t = torch.arange(length)[:, None] * inv_tscales[None, :]
    pos1 = torch.sin(scaled_t)
    pos2 = torch.cos(scaled_t)
    positions = torch.cat([pos1, pos2], dim=1)  
    return nn.Parameter(positions.clone())

def accumulate_phase_mod(f0, t_frame, phi0=0.0):
    omega = 2 * torch.pi * f0
    dphi = omega * t_frame
    phi = torch.cumsum(dphi, dim=0) + phi0
    phi = torch.remainder(phi, 2 * torch.pi)
    return phi

class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, radii=True, debug: List[str] = [], use_pbias=False, axial=False, spec_shape=None, relative=False, freq_bins=None):
        super(rotary, self).__init__()
        self.use_pbias = use_pbias
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.radii = radii
        self.debug = debug
        self.counter = 0
      
        self.axial = axial
        if axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer('time_freqs', time_freqs)
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer('freq_freqs', freq_freqs)

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2), requires_grad=True if use_pbias else False)
        theta = (torch.tensor(10000, device=device, dtype=dtype))
        self.theta = nn.Parameter(theta, requires_grad=True)    
        self.theta_values = []

        self.relative = relative
        self.freq_bins = freq_bins
        self.true2d_dim = (dims // head) // 2
        self.omega_t = nn.Parameter(torch.randn(self.true2d_dim))
        self.omega_f = nn.Parameter(torch.randn(self.true2d_dim))

    def axial(self, seq_len):
        if not self.use_2d_axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
        t = torch.arange(seq_len, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

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

    def accumulate_phase_mod(self, f0, t_frame, phi0=0.0):
        omega = 2 * torch.pi * f0
        dphi = omega * t_frame
        phi = torch.cumsum(dphi, dim=0) + phi0
        phi = torch.remainder(phi, 2 * torch.pi) 
        return phi

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
        if f0 is not None and f0.dim() == 2:
            f0 = f0.squeeze(0) 
        if f0t is not None and f0t.dim() == 2:
            f0t = f0t.squeeze(0) 
        if f0 is not None and f0.shape[0] == ctx:
            return f0
        elif f0t is not None and f0t.shape[0] == ctx:
            return f0t
        else:
            return None         

    def forward(self, x=None, enc=None, layer=None, feature=None) -> Tensor:
        ctx = x
        if self.axial and feature == "spectrogram":
            freqs_2d = self.axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)

        f0 = enc.get("f0") if enc is not None else None 
        f0t = enc.get("f0t") if enc is not None else None 
        f0 = self.check_f0(f0, f0t, ctx)
        theta = f0 + self.theta if f0 is not None else self.theta
        # theta = f0
        freqs = self.theta_freqs(theta)
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = t[:, None] * freqs
        freqs, radius = self._apply_radii(freqs, f0, ctx)
        
        if "radius" in self.debug and self.counter == 10:
            print(f"  [{layer}] [Radius] {radius.shape if radius is not None else None} {radius.mean() if radius is not None else None} [Theta] {theta.mean() if theta is not None else None} [f0] {f0.shape if f0 is not None else None}  [ctx] {ctx}")

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


    # @staticmethod
    # def apply_rotary(x, freqs):
    #     # x: [batch, head, seq, head_dim]
    #     # freqs: [1, seq, head_dim] or [1, seq, 2*head_dim] for 2D
    #     if freqs.shape[-1] == x.shape[-1]:
    #         # 1D rotary
    #         x1 = x
    #         orig_shape = x1.shape
    #         if x1.ndim == 2:
    #             x1 = x1.unsqueeze(0)
    #         x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
    #         x1 = torch.view_as_complex(x1) * freqs
    #         x1 = torch.view_as_real(x1).flatten(-2)
    #         x1 = x1.view(orig_shape)
    #         return x1.type_as(x)
    #     else:
    #         # 2D rotary: split x and apply to each axis
    #         head_dim = x.shape[-1] // 2
    #         x_time = x[..., :head_dim]
    #         x_freq = x[..., head_dim:]
    #         f_time = freqs[..., :head_dim]
    #         f_freq = freqs[..., head_dim:]
    #         # Apply rotary to each axis
    #         def apply_axis(xa, freqs):
    #             orig_shape = xa.shape
    #             xa = xa.float().reshape(*xa.shape[:-1], -1, 2).contiguous()
    #             xa = torch.view_as_complex(xa) * freqs
    #             xa = torch.view_as_real(xa).flatten(-2)
    #             xa = xa.view(orig_shape)
    #             return xa.type_as(x)
    #         x_time = apply_axis(x_time, f_time)
    #         x_freq = apply_axis(x_freq, f_freq)
    #         return torch.cat([x_time, x_freq], dim=-1)

    # def true2d_relative_angle(self, t_q, f_q, t_k, f_k):
    #     # t_q, f_q, t_k, f_k: [seq]
    #     delta_t = t_q[:, None] - t_k[None, :]  # [seq, seq]
    #     delta_f = f_q[:, None] - f_k[None, :]  # [seq, seq]
    #     angle = delta_t[..., None] * self.omega_t + delta_f[..., None] * self.omega_f  # [seq, seq, true2d_dim]
    #     angle = torch.cat([angle, angle], dim=-1)  # [seq, seq, head_dim]
    #     return angle

    # def true2d_apply_rotary(self, q, k, freqs):
    #     # q, k: [batch, head, seq, head_dim]
    #     # freqs: [seq, seq, head_dim//2] complex, or [seq, seq, head_dim] if you want
    #     b, h, seq, d = q.shape
    #     d2 = d // 2
    #     q_exp = q.unsqueeze(3).expand(b, h, seq, seq, d)
    #     k_exp = k.unsqueeze(2).expand(b, h, seq, seq, d)
    #     # Convert to complex
    #     def to_complex(x):
    #         return torch.complex(x[..., 0::2], x[..., 1::2])  # [b, h, seq, seq, d2]
    #     q_c = to_complex(q_exp)
    #     k_c = to_complex(k_exp)
    #     # Multiply by freqs (which should be [seq, seq, d2] complex)
    #     q_rot = q_c * freqs
    #     k_rot = k_c * freqs
    #     # Back to real
    #     def to_real(x):
    #         return torch.stack([x.real, x.imag], dim=-1).flatten(-2)
    #     q_rot = to_real(q_rot)
    #     k_rot = to_real(k_rot)
    #     return q_rot, k_rot


def parallel_slice(self, q, k, v, mask=None):
    batch, head, ctx, dims = q.shape
    head_dim = self.head_dim
    batch, ctx, dims = q.shape
    ctx_len = k.shape[1]
    head = dims // head_dim
    scores = torch.zeros(batch, head, ctx, ctx_len, device=q.device)
    for h in range(head):
        start_idx = h * head_dim
        end_idx = start_idx + head_dim
        q_h = q[:, :, start_idx:end_idx]
        k_h = k[:, :, start_idx:end_idx]
        scores[:, h] = torch.bmm(q_h, k_h.transpose(1, 2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.zeros_like(q)
    for h in range(head):
        start_idx = h * head_dim
        end_idx = start_idx + head_dim
        v_h = v[:, :, start_idx:end_idx]
        output[:, :, start_idx:end_idx] = torch.bmm(attn_weights[:, h], v_h)
    return output    

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

class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        self.head  = head
        self.hdim  = dims // head
        self.scale = scale                      
        self.q_proj = Linear(dims, dims)
        self.k_proj = Linear(dims, dims)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor | None:
        B, Q, _ = x.shape
        K       = guide.size(1)
        q = self.q_proj(x ).view(B, Q, self.head, self.hdim).transpose(1,2)
        k = self.k_proj(guide).view(B, K, self.head, self.hdim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.hdim)
        return bias

class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 1e-7, minz: float = 1e-8, maxz: float = 1e-6, debug: List[str] = [], use_pbias=False, relative=False, freq_bins=None, radii=True, axial=False, spec_shape=None, rbf=False):
                 
        super(MultiheadA, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self.counter = 0
        self.use_pbias = use_pbias
        self.relative = relative
        self.freq_bins = freq_bins
        self.rbf = rbf

        self.q = nn.Linear(dims, dims).to(device, dtype)
        self.k = nn.Linear(dims, dims, bias=False).to(device, dtype)
        self.v = nn.Linear(dims, dims).to(device, dtype)
        self.o = nn.Linear(dims, dims).to(device, dtype)

        self.pad_token = 0
        self.rotary_emb = rotary_emb
        self.minz = minz
        self.maxz = maxz
        self.zero_val = zero_val   
        self.fzero = nn.Parameter(torch.tensor(zero_val, device=device, dtype=dtype), requires_grad=False)
        
        if rotary_emb:
            self.rope = rotary(
                dims=dims,
                head=head,
                debug=debug,
                radii=radii,
                relative=relative,
                freq_bins=freq_bins,
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
          
    def forward(self, x: Tensor, xa = None, mask = None, enc = None, layer = None, feature=None) -> tuple:

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

            if self.relative and feature == "spectrogram":
                seq_len = q2
                freq_bins = self.freq_bins
                idxs = torch.arange(seq_len, device=q.device)
                t_idx = idxs // freq_bins
                f_idx = idxs % freq_bins
                angle = self.rope.relative(t_idx, f_idx, t_idx, f_idx)
                q_rot, k_rot = self.rope.d2rotary(q, k, angle)
                scale = (self.dims // self.head) ** -0.25
                qk = (q_rot * scale * k_rot * scale).sum(-1)
                w = F.softmax(qk, dim=-1).to(q.dtype)
                wv = torch.einsum('bhij,bhjd->bhid', w, v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1))
                wv = wv.permute(0, 2, 1, 3).flatten(start_dim=2)
                return self.o(wv), qk
            else:
                q = self.rope.apply_rotary(q, (self.rope(x=q2, enc=enc, layer=layer, feature=feature)))
                k = self.rope.apply_rotary(k, (self.rope(x=k2, enc=enc, layer=layer, feature=feature)))
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if self.rbf:
            qk = self.rbf_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        if self.use_pbias:
            pbias = self.rope.pitch_bias(f0 = enc.get("f0", None) if enc is not None else None) 
            if pbias is not None:
                qk = qk + pbias[:,:,:q2,:q2]

        if mask is not None:
            mask = mask[:q2, :q2]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if xa is not None:
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
    def __init__(self, dims, head, enabled=True, one_shot=False):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

        if one_shot:
            self.one_shot = OneShot(dims, head)
    
    def forward(self, x, xa=None):
        if not self.enabled:
            return None
        if self.one_shot:
            x = self.one_shot(x, xa)
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
        self.one_shot = OneShot(dims, head) if one_shot else None

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

    def forward(self, x, xa=None, mask=None, enc=None, layer=None, feature=None) -> Tensor:
 
        b = torch.sigmoid(self.blend)
        ax = x + self.attn(self.lna(x), xa=xa, mask=mask, enc=enc, layer=layer, feature=feature)[0]
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
                    dims=dims,
                    head=head,
                    use_2d_axial=True,
                    spec_shape=spec_shape, debug=[])
            else:
                self.rope = rotary(
                    dims=dims,
                    head=head,
                    use_2d_axial=False, debug=[])
        else:
            self.rope = None
            self.sinusoid_pos = lambda length, dims: sinusoids(length, dims, max_tscale=10000)
            
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer="FEncoder", feature="spectrogram"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        if feature == "spectrogram" and self.rope is not None:
            rope_freqs = self.rope(ctx, layer=layer, feature="spectrogram")
        else:
            rope_freqs = self.rope(ctx, layer=layer, feature="audio")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x

    def forward(self, x, enc=None, feature="spectrogram", layer="FEncoder"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, layer=layer, feature=feature)
        else:
            x = x + self.sinusoid_pos(x.shape[1], x.shape[-1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

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
            self.sinusoid_pos = lambda length, dims: sinusoids(length, dims, max_tscale=10000)   
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer="WEncoder", feature="waveform"):
        if not self.use_rope or self.rope is None:
            return x
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(ctx, layer=layer, feature=feature)
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, enc=None, feature="waveform", layer="WEncoder"):
        x = self.downsample(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, layer=layer)
        else:
            x = x + self.sinusoid_pos(x.shape[1], x.shape[-1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False, one_shot=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        self.one_shot = one_shot
        act_fn = get_activation(act)

        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)

        
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                head=self.head,
                debug=[])
        else:
            self.rope = None
            self.sinusoid_pos = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, layer="PEncoder", feature="pitch"):
        if not self.use_rope or self.rope is None:
            return x
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(ctx, layer=layer, feature=feature)
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, xa, enc=None, layer="PEncoder", feature="pitch"):
        xa = self.encoder(xa).permute(0, 2, 1)
        if self.use_rope:
            xa = self.apply_rope_to_features(xa, layer=layer)
        else:
            xa = xa + self.sinusoid_pos(xa.shape[1], xa.shape[-1], 10000).to(xa.device, xa.dtype)
        if self.one_shot:
            x = enc["input_ids"]
            xa = self.one_shot(x, xa)
        xa = nn.functional.dropout(xa, p=self.dropout, training=self.training)
        return self.norm(xa)

def win_mask(text_ctx, aud_ctx):
    mask = torch.tril(torch.ones(text_ctx, text_ctx, device=device), diagonal=0)
    audio_mask = torch.tril(torch.ones(text_ctx, aud_ctx - text_ctx, device=device))
    full_mask = torch.cat([mask, audio_mask], dim=-1)
    return full_mask.unsqueeze(0).unsqueeze(0)

def causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0).unsqueeze(0).unsqueeze(0)

class theBridge(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, 
                debug: List[str], features: List[str], act: str = "gelu"): 
        super(theBridge, self).__init__()
    
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

        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
        self.ln_dec = RMSNorm(dims)
        self.sinusoid_pos = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)

        with torch.no_grad():
            self.token.weight[0].zero_()
        
        self.block = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", debug=debug, features=features)
            for _ in range(layer)])  

        self.cross_attn = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", debug=debug, features=features)
            for _ in range(layer)])  

        self.cross_modal = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", debug=debug, features=features)
            for _ in range(layer)])  
        
        self.register_buffer("mask", causal_mask(ctx, device), persistent=False)
        self.register_buffer("mask_win", win_mask(ctx, ctx), persistent=False)

        act_fn = get_activation(act)
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
            
        self.blockA = nn.ModuleDict({
            "spectrogram": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "spectrogram" in features else None), 
            "waveform": nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "waveform" in features else None),
            "pitch": nn.ModuleList(
            [PEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=3, act=act, one_shot=False)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "pitch" in features else None),
            "envelope": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)]  if "envelope" in features else None),
            "phase": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "phase" in features else None)})



    def forward(self, x, enc, feature, layer='theBridge') -> Tensor:
        f0 = enc.get("f0")
        out = {}
        out.update(enc)
        enc = dict_to(enc, device, dtype)
        _text_len = x.shape[1]  
        x = self.token(x) + self.positional[:x.shape[1]]
   
        for f in enc:
            if f in self.features:
                xa = enc[f]
                for block in self.blockA[f]:
                    xa = block(xa, enc=out, feature=feature, layer="enc_self")
                    xa = xa + self.sinusoid_pos(xa.shape[1], xa.shape[-1], 10000).to(xa.device, xa.dtype)
                    out[f] = xa

        for block in self.block:          
            x = block(x, xa=None, mask=self.mask, enc=enc, feature=feature, layer="dec_self")
            out["input_ids"] = x

            if f in self.features:
                out = block(x, xa=xa, mask=self.mask, enc=enc, feature=feature, layer="dec_cross")
                if self.sequential:
                    x = out
                else:
                    a = torch.sigmoid(self.blend)
                    x = a * out + (1 - a) * x
                    x = self.token(x) + self.positional[:x.shape[1]]
                    out[f] = x

        for block in self.cross_attn:
            if f in self.features:
                x = block(x, xa=xa, mask=self.mask, enc=enc, feature=feature, layer="dec_cross")
                xa = block(xa, xa=x, mask=self.mask, enc=enc, feature=feature, layer="enc_cross")
                out = block(x, xa=xa, mask=self.mask, enc=enc, feature=feature, layer="dec_cross")
                if self.sequential:
                    x = out
                else:
                    a = torch.sigmoid(self.blend)
                    x = a * out + (1 - a) * x
                    x = self.token(x) + self.positional[:x.shape[1]]
                    out[f] = x

        for block in self.cross_modal:
            if f in self.features:
                xcat = torch.cat([x, xa], dim=1)
                x = block(xcat, xa=None, mask=self.mask, enc=enc, feature=feature, layer="cross_modal")
                x = x[:, :_text_len]  
                out[f] = x

        if self.counter < 1 and "encoder" in self.debug:      
                shapes = {k: v.shape for k, v in enc.items()}
                print(f"Step {self.counter}: mode: {list(enc.keys()) }: shapes: {shapes}")
        self.counter += 1
       
        x = self.ln_dec(x)
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()
        return x, out

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
        ) -> Dict[str, Optional[torch.Tensor]]:

        enc = {}
        if spectrogram is not None:
            enc["spectrogram"] = spectrogram
            feature = "spectrogram"
        if waveform is not None:
            enc["waveform"] = waveform
            feature = "waveform"
        if pitch is not None:
            enc["pitch"] = pitch
            feature = "pitch"
        if f0 is not None:
            enc["f0"] = f0
        if f0t is not None:
            enc["f0t"] = f0t
        if harmonic is not None:
            enc["harmonic"] = harmonic
        if aperiodic is not None:
            enc["aperiodic"] = aperiodic
        if input_ids is not None:
            enc["input_ids"] = input_ids
            feature = "input_ids"

        logits, out = self.processor(input_ids, enc, feature)
        self.out = out

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

    def bdec(ids_list, skip_special_tokens=True, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        results = []
        for ids in ids_list:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = [int(id) for id in ids if id != -100]
            if skip_special_tokens:
                ids = [id for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id)]

                if ids and ids and ids[0] == bos_token_id:
                    ids = ids[1:]
                while ids and ids[-1] == eos_token_id:
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

def tokenize_pitch(pitch_features, target_length):
    pitch_len = pitch_features.shape[-1]
    token_len = target_length
    if pitch_len > token_len:
        pitch_tokens = F.adaptive_avg_pool1d(pitch_features, token_len)
    else:
        pitch_tokens = F.interpolate(pitch_features, token_len)
    return pitch_tokens

def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
          
    return waveform

def world_to_mel(sp, ap, sample_rate=16000, n_mels=128):
    import librosa
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()
    sp_mel = torch.matmul(sp, mel_basis.T)
    ap_mel = torch.matmul(ap, mel_basis.T)
    return sp_mel, ap_mel

def extract_features(batch, tokenizer, waveform=False, spec=False, f0=True, f0t=True, pitch=True, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, **dataset_config):
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
        "normalized": False,
    }

    audio = batch["audio"]
    sr = audio["sampling_rate"]
    labels = tokenizer.encode(batch["transcription"])

    wav = wavnp = f0_np = t = None
    spectrogram = f0_tensor = f0t_tensor = harmonic = aperiodic = p_tensor = None

    if waveform or spec or f0 or f0t or harmonics or pitch:
        wav = load_wave(wave_data=audio, sample_rate=sr)
        wavnp = wav.numpy().astype(np.float64)

    if spec:
        transform = torchaudio.transforms.MelSpectrogram(**dataset_config)
        mel_spectrogram = transform(wav)
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spectrogram = (log_mel + 4.0) / 4.0
        spectrogram = torch.tensor(spectrogram)

    if f0 or f0t or harmonics or pitch:
        f0_np, t = pw.dio(wavnp, sample_rate,
        frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)
        t = torch.tensor(t)

    if f0:
        f0_tensor = torch.from_numpy(f0_np)
        # t_frame = torch.mean(t[1:] - t[:-1])
        # f0_tensor = accumulate_phase_mod(f0_tensor, t_frame)
 
    if f0t:
        audio_duration = len(wavnp) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t, token_starts, side="left")
        end_idx = torch.searchsorted(t, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)
        for i in range(T):
            lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i])
            segment = f0_np[lo:hi]
            if mode == "mean":
                pitch_tok[i] = segment.mean()
            elif mode == "median":
                pitch_tok[i] = torch.median(segment)
            else:
                pitch_tok[i] = segment[-1]
        pitch_tok[pitch_tok < 100.0] = 0.0
        bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
        f0t_tensor = torch.from_numpy(np.concatenate([[bos_pitch], pitch_tok]))
        # f0t_tensor = accumulate_phase_mod(f0t_tensor, t_frame)

    if pitch:
        p_tensor = torch.from_numpy(f0_np)
        p_tensor = p_tensor.unsqueeze(0)

    if harmonics:
        spnp = pw.cheaptrick(wavnp, f0_np, t, sample_rate, fft_size=256)
        apnp = pw.d4c(wavnp, f0_np, t, sample_rate, fft_size=256)
        harmonic = torch.from_numpy(spnp)
        aperiodic = torch.from_numpy(apnp)
        harmonic = harmonic[:, :128].contiguous().T
        aperiodic = aperiodic[:, :128].contiguous().T
        harmonic = torch.where(harmonic == 0.0, torch.zeros_like(harmonic), harmonic / 1.0)
        aperiodic = torch.where(aperiodic == 0.0, torch.zeros_like(aperiodic), aperiodic / 1.0)

    if debug:
        print(f"['f0']: {f0_tensor.shape if f0 is not None else None}")
        print(f"['f0t']: {f0t_tensor.shape if f0t is not None else None}")
        print(f"['harmonic']: {harmonic.shape if harmonic is not None else None}")
        print(f"['aperiodic']: {aperiodic.shape if aperiodic is not None else None}")
        print(f"['spectrogram']: {spectrogram.shape if spectrogram is not None else None}")
        print(f"['waveform']: {wav.shape if wav is not None else None}")
        print(f"['labels']: {len(labels) if labels is not None else None}")

    return {
        "waveform": wav if waveform else None,
        "spectrogram": spectrogram if spec else None,
        "f0": f0_tensor if f0 else None,
        "f0t": f0t_tensor if f0t else None,
        "pitch": p_tensor if pitch else None,
        "harmonic": harmonic if harmonics else None,
        "aperiodic": aperiodic if harmonics else None,
        "labels": labels,
    }

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False, **dataset_config):

        if sanity_check:
            test = load_dataset(
                "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True
            ).cast_column("audio", Audio(sampling_rate=sample_rate)).take(10)
            dataset = test.map(
                lambda x: extract_features(x, tokenizer, **dataset_config),
                remove_columns=test.column_names)

            train_dataset = dataset
            test_dataset = dataset
            return train_dataset, test_dataset
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
            return (0 < len(x["transcription"]) < 2048 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 2048 * 160)

        raw_train = load_dataset(
            "google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=streaming).take(1000)
        raw_test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).take(100)

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

        train_dataset.save_to_disk(cache_file_train) if sanity_check is False else None
        test_dataset.save_to_disk(cache_file_test) if sanity_check is False else None
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

            elif key in ["spectrogram", "waveform", "pitch", "harmonic", "aperiodic", "f0t", "f0"]:
                items = [f[key] for f in features if key in f]
                items = [item for item in items if item is not None]
                if not items:  
                    continue
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

def clean_ids(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(id) for id in ids if id != -100 and id != pad_token_id and id != bos_token_id and id != eos_token_id]

def clean_batch(batch_ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    return [clean_ids(seq, pad_token_id, bos_token_id, eos_token_id) for seq in batch_ids]

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, optimizer=None, scheduler=None):

    label_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    label_ids = clean_batch(label_ids, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    pred_ids = clean_batch(pred_ids, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    if print_pred:
        for i in range(min(num_samples, len(pred_ids))):
            print(f"Pred tokens: {pred_ids[i]}")
            print(f"Label tokens: {label_ids[i]}")
            print(f"Pred: '{pred_str[i]}'")
            print(f"Label: '{label_str[i]}'")
            print("-" * 40)

    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
    else:
        trainable_params = 0.0
        efficiency_score = 0.0
    return {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    labels = torch.where(labels == -100, 0, labels)
    pred_ids = torch.where(pred_ids == -100, 0, pred_ids)

    return pred_ids, labels

def main():
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)
    train_dataset, test_dataset = prepare_datasets(
    tokenizer, 
    token, 
    sanity_check=True,

    )
    
    param = Dimensions(
        vocab=40000,
        mels=128,
        ctx=1500,
        dims=512,
        head=4,
        layer=4,
        act="swish",
        debug={"radius", "encoder"},
        features = ["pitch"],
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
        save_strategy="no",
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        eval_on_start=True,
        batch_eval_metrics=False,
    )
    from functools import partial
    metrics_fn = partial(compute_metrics, 
    print_pred=False, 
    num_samples=2, 
    tokenizer=tokenizer, model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, eps=1e-8, weight_decay=0.025, betas=(0.9, 0.999), 
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

