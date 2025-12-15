import torch
import os
import pyworld as pw
from einops import rearrange, pack, unpack, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple, Callable
import math
from math import sqrt
import matplotlib.pyplot as plt
from torch import nn, einsum
import torch.nn.init as init
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from functools import lru_cache
from subprocess import CalledProcessError, run
from math import gcd
from collections import namedtuple
from functools import partial, reduce

from functools import wraps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def causal_mask(q, k, device):
    return torch.ones((q, k), device = device, dtype = torch.bool).triu(k - q + 1)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def valid(default_value, *items):
    for item in items:
        if item is not None:
            return item
    return default_value

def maybe(fn):
    if not have(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not have(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def dict_to(d, device, dtype=dtype):
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}  

def have(a):
    return a is not None
    
def Sequential(*modules):
    return nn.Sequential(*filter(have, modules))    

def AorB(a, b):
    return a if have(a) else b

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    if have(mask):
        t = t.masked_fill(~mask[..., None], 0.)
    return F.pad(t, (0, 0, amount, -amount), value = 0.)

def always(value):
    return lambda *args, **kwargs: value

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val
        
def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

class FourierTransformerGenerator(nn.Module):
    def __init__(n, 
                 n_mels=80, 
                 d_model=512, 
                 nhead=8, 
                 num_layers=4, 
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024):

        super().__init__()
        n.d_model = d_model
        n.n_fft = n_fft
        n.hop_length = hop_length
        n.win_length = win_length

        n.input_projection = nn.Linear(n_mels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        n.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        n.output_projection = nn.Linear(d_model, (n_fft // 2 + 1) * 2)

    def forward(n, mel_spectrogram):

        mel_spectrogram = mel_spectrogram.transpose(1, 2)
        x = n.input_projection(mel_spectrogram)
        x = n.transformer_encoder(x)
        fft_coeffs = n.output_projection(x)
        b, t, _ = fft_coeffs.shape
        fft_coeffs = fft_coeffs.view(b, t, n.n_fft // 2 + 1, 2)
        real_part, imag_part = fft_coeffs.unbind(dim=-1)
        complex_spectrogram = torch.complex(real_part, imag_part)
        
        waveform = torch.istft(
            complex_spectrogram.transpose(1, 2),
            n_fft=n.n_fft,
            hop_length=n.hop_length,
            win_length=n.win_length,
            window=torch.hann_window(n.win_length, device=complex_spectrogram.device))
        return waveform

class DSPExcitationGenerator(nn.Module):
    def __init__(n, sample_rate, hop_length):
        super().__init__()
        n.sample_rate = sample_rate
        n.hop_length = hop_length
        n.noise_gain = 0.003

    def forward(n, f0, aperiodicity):
        batch_size, seq_len = f0.shape
        audio_len = seq_len * n.hop_length
        
        excitation = torch.zeros(batch_size, audio_len, device=f0.device)
        
        for b in range(batch_size):
            f0_seq = f0[b]
            voiced_frames = f0_seq > 0
            voiced_f0 = f0_seq[voiced_frames]

            if voiced_f0.numel() > 0:
                phase = 0.0
                for q in range(seq_len):
                    if f0_seq[q] > 0:
                        samples_per_cycle = n.sample_rate / f0_seq[q]
                        while phase < n.hop_length:
                            pos = int(q * n.hop_length + phase)
                            if pos < audio_len:
                                excitation[b, pos] = 1.0
                            phase += samples_per_cycle
                        phase -= n.hop_length
            
            noise = torch.randn(audio_len, device=f0.device) * n.noise_gain
            excitation[b, :] += noise
        return excitation.unsqueeze(1)

class F0TransformerVocoder(nn.Module):
    def __init__(n, 
                 n_mels=80, 
                 d_model=512, 
                 nhead=8, 
                 num_layers=4, 
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024,
                 sample_rate=22050):
        super().__init__()
        n.d_model = d_model
        n.hop_length = hop_length
        n.sample_rate = sample_rate

        n.input_projection = nn.Linear(n_mels + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        n.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        n.excitation_generator = DSPExcitationGenerator(sample_rate, hop_length)
        n.filter_projection = nn.Linear(d_model, 128)
        n.upsample = nn.Upsample(scale_factor=hop_length, mode='nearest')
        
    def forward(n, mel_spectrogram, f0, aperiodicity=None):
        mel_spectrogram = mel_spectrogram.transpose(1, 2)
        f0 = f0.transpose(1, 2)
        x = torch.cat([mel_spectrogram, f0], dim=-1)
        
        x = n.input_projection(x)
        x = n.transformer_encoder(x)
        filter_coeffs = n.filter_projection(x)
        excitation = n.excitation_generator(f0.squeeze(-1), aperiodicity)
        filter_coeffs = filter_coeffs.transpose(1, 2)
        upsampled_filters = n.upsample(filter_coeffs)
        
        waveform = F.conv1d(
            excitation, 
            upsampled_filters,
            padding='same',
            groups=excitation.shape[0])
        return waveform.squeeze(1)

class AudioDiffusionTransformer(nn.Module):
    def __init__(n,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 input_size: int = 16000,
                 vocab_size: int = 1000):

        super().__init__()
        n.d_model = d_model
        n.input_size = input_size
        n.vocab_size = vocab_size
        n.audio_embed = nn.Linear(input_size, d_model)
        n.timestep_embed = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        n.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        n.output_projection = nn.Linear(d_model, input_size)

    def forward(n, noisy_audio: torch.Tensor, timesteps: torch.Tensor):
        batch_size = noisy_audio.shape[0]
        audio_embedded = n.audio_embed(noisy_audio)
        timestep_embedded = n.timestep_embed(timesteps)
        combined_embeddings = torch.stack([audio_embedded, timestep_embedded], dim=1)
        transformer_output = n.transformer_encoder(combined_embeddings)
        noise_prediction = n.output_projection(transformer_output[:, 0, :])
        return noise_prediction

class AdaLayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.parameter_generator = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, dims * 2))
        self.norm = nn.LayerNorm(dims, elementwise_affine=False)

    def forward(self, x):
        sequence_length = x.shape[-1]
        timesteps = torch.arange(sequence_length, device=device).unsqueeze(0).expand(x.shape)
        sinusoidal_emb = self.sinusoidal_encoding(timesteps, self.dims)
        emb = self.parameter_generator(sinusoidal_emb)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x

    def sinusoidal_encoding(self, timesteps, dims, max_tscale=10000.0):
        q = torch.arange(0, dims, 2).float().to(timesteps.device)
        frequencies = torch.exp(q * -math.log(max_tscale) / dims)
        timesteps_expanded = timesteps.unsqueeze(-1).float()
        pe = torch.zeros_like(timesteps_expanded.expand(-1, -1, dims))
        pe[..., 0::2] = torch.sin(timesteps_expanded * frequencies)
        pe[..., 1::2] = torch.cos(timesteps_expanded * frequencies)
        return pe

# class AdaLayerNorm(nn.Module):
#     def __init__(n, num_embeddings: int, eps: float = 1e-6):
#         super().__init__()
#         dims = num_embeddings
#         n.eps = eps
#         n.dim = dims
#         n.scale = nn.Embedding(num_embeddings=num_embeddings, dims=dims)
#         n.shift = nn.Embedding(num_embeddings=num_embeddings, dims=dims)
#         torch.nn.init.ones_(n.scale.weight)
#         torch.nn.init.zeros_(n.shift.weight)

#     def forward(n, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
#         scale = n.scale(cond_embedding_id)
#         shift = n.shift(cond_embedding_id)
#         x = nn.functional.layer_norm(x, (n.dim,), eps=n.eps)
#         x = x * scale + shift
#         return x

class LayerNorm(nn.Module):
    def __init__(n, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(LayerNorm, n).__init__()
        if isinstance(dims, int):
            n.normalized_shape = (dims,)
        else:
            n.normalized_shape = tuple(dims)
        n.eps = eps
        n.elementwise_affine = elementwise_affine
        if n.elementwise_affine:
            n.weight = nn.Parameter(torch.empty(n.normalized_shape))  # type: ignore
            torch.nn.init.ones_(n.weight)  
        else:
            n.register_parameter("weight", None)
    def forward(n, x):
        return F.rms_norm(x, n.normalized_shape, n.weight, n.eps)  # type: ignore

class L4TextRMSNorm(nn.Module):
    def __init__(self, x, eps=1e-5):

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(x))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

# class LayerNorma(nn.Module):
#     def __init__(n, dims):
#         super().__init__()
#         n.eps = 1e-5
#         n.scale = nn.Parameter(torch.ones(dims))
#         n.shift = nn.Parameter(torch.zeros(dims))

#     def forward(n, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True, unbiased=False)
#         norm_x = (x - mean) / torch.sqrt(var + n.eps)
#         return n.scale * norm_x + n.shift

# class LayerNormb(nn.Module):
#     def __init__(n, dims):
#         super().__init__()
#         n.gamma = nn.Parameter(torch.ones(dims))
#         n.register_buffer("beta", torch.zeros(dims))

#     def forward(n, x):
#         return F.layer_norm(x, x.shape[-1:], n.gamma, n.beta)

# class RMSNorm(nn.Module):
#     def __init__(n, dims, eps=1e-6):
#         super().__init__()
#         n.scale = dims ** 0.5
#         n.gamma = nn.Parameter(torch.ones(dims))

#     def forward(n, x):
#         return n.gamma * F.normalize(x, dim = -1) * n.scale

# class RMSNorman(nn.Module):
#     def __init__(n, dims, eps=1e-6):
#         super().__init__()
#         n.dims = dims
#         n.eps = eps
#         n.gamma = nn.Parameter(torch.ones(dims))
        
#     def forward(n, x):
#         rms = torch.sqrt(torch.mean(x**2, dim=2, keepdim=True) + n.eps)
#         normalized_x = x / rms
#         gamma_reshaped = n.gamma.view(1, -1, 1)
#         output = normalized_x * gamma_reshaped
#         return output

class InstanceRMS(nn.Module):
    def __init__(n, dims, eps=1e-6):
        super().__init__()
        n.instance_norm = nn.InstanceNorm1d(
            dims,
            eps=eps,
            affine=False,
            track_running_stats=False
        )
        n.gamma = nn.Parameter(torch.ones(dims))

    def forward(n, x):
        normalized_x = n.instance_norm(x)
        gamma_reshaped = n.gamma.view(1, -1, 1)
        output = normalized_x * gamma_reshaped
        return output

# class RMSNormandy(nn.Module):
#     def __init__(n, dims):
#         super().__init__()
#         n.scale = dims // 2
#         n.gamma = nn.Parameter(torch.ones(dims))
#         n.normalized_shape = (dims,)
#         n.weight = nn.Parameter(torch.empty(n.normalized_shape))
#         n.register_buffer("beta", torch.zeros(dims))

#     def forward(n, x):
#         return n.gamma * F.layer_norm(x, n.normalized_shape, n.gamma, n.beta) * n.scale        

# def get_norm(norm_type: str, dims, num_groups) -> nn.Module:

#     norm_map = {
#         "layernorm": nn.LayerNorm([dims]),
#         "instancenorm1d": nn.InstanceNorm1d(dims),
#         "batchnorm1d": nn.BatchNorm1d(dims),
#         "groupnorm": nn.GroupNorm(num_groups, dims),
#         "rmsnorm": RMSNorm([dims]),
#     }
    
#     return norm_map.get(norm_type.lower(), nn.LayerNorm([dims]))

def get_norm(norm_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if norm_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{norm_type}' requires 'dims'.")
    if norm_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{norm_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        "AdaLayerNorm": lambda: AdaLayerNorm(dims=dims),
        "layernormB": lambda: LayerNorm(dims=dims),
        "InstanceRMS": lambda: InstanceRMS(dims=dims),        
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=dims),
        "rmsnormB": lambda: RMSNorm(dims=dims),
        }
   
    norm_func = norm_map.get(norm_type)
    if norm_func:
        return norm_func()
    else:
        print(f"Warning: Norm type '{norm_type}' not found. Returning LayerNorm.")
        return nn.LayerNorm(dims) 

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32 if torch.cuda.is_available() else torch.float64

def tox():
    return {"device": get_device(), "dtype": get_dtype()}

def cos_xy(x: Tensor, y: Tensor) -> Tensor:
    out = F.softmax(torch.matmul(F.normalize(x, dim=-1), F.normalize(y, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
    return out

def cos_qkv(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    out = torch.matmul(F.softmax(torch.matmul(torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1).transpose(-1, -2)) + (mask if mask is not None else 0), dim=-1), v)
    return out

def cos_qkv_scaled(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    dk = q.size(-1)
    attn_scores = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
    scaled_attn_scores = attn_scores / math.sqrt(dk)
    if mask is not None:
        scaled_attn_scores = scaled_attn_scores + mask 
    attention_weights = F.softmax(scaled_attn_scores, dim=-1)
    out = torch.matmul(attention_weights, v)
    return out

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(n, channels):
        super().__init__()
        n.alpha = nn.Parameter(torch.ones(1, 1, channels))

    def forward(n, x):
        return snake(x, n.alpha)

class PeriodicReLU(nn.Module):
    def __init__(n, period=1.0, slope=1.0, bias=0.0):
        super().__init__()

        n.period = nn.Parameter(torch.tensor(period))
        n.slope = nn.Parameter(torch.tensor(slope))
        n.bias = nn.Parameter(torch.tensor(bias))

    def forward(n, x):
        scaled_x = x * (math.pi / n.period)
        sawtooth = scaled_x - torch.floor(scaled_x)
        triangle_wave = 2 * torch.abs(sawtooth - 0.5) - 1
        return n.slope * triangle_wave + n.bias

def get_activation(act: str) -> nn.Module:

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
        "elu": nn.ELU(),
        "PeriodicReLU": PeriodicReLU(),
        # "snake": Snake1d,
    }
    return act_map.get(act, nn.GELU())

class RelativePositionBias(nn.Module):
    def __init__(n, dims, head, layer = 3):
        super().__init__()
        n.net = nn.ModuleList([])
        n.net.append(nn.Sequential(nn.Linear(1, dims), nn.SiLU()))

        for _ in range(layer - 1):
            n.net.append(nn.Sequential(nn.Linear(dims, dims), nn.SiLU()))
        n.net.append(nn.Linear(dims, head))

    @property
    def device(n):
        return next(n.parameters()).device

    def forward(n, q, k):
        assert k >= q
        device = n.device

        q_pos = torch.arange(q, device = device) + (k - q)
        k_pos = torch.arange(k, device = device)
        rel_pos = (rearrange(q_pos, 'q -> q 1') - rearrange(k_pos, 'k -> 1 k'))
        rel_pos += (k - 1)
        x = torch.arange(-k + 1, k, device = device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in n.net:
            x = layer(x)
        x = x[rel_pos]
        return rearrange(x, 'q k h -> h q k')

def sinusoids(ctx, dims, theta=10000):
    # if isinstance(theta, int):
    #     print(theta)
    # elif isinstance(theta, torch.Tensor):    
 
    #     print(theta.shape)
    # print(theta)
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding

class AbsolutePositions(nn.Module):
    def __init__(n, ctx, dims):
        super().__init__()
        n.emb = nn.Embedding(ctx, dims)
    def forward(n, x):
        return n.emb(torch.arange(x.shape[1], device=device))[None, :, :]

class FixedPositions(nn.Module):
    def __init__(n, dims, head, ctx, theta=10000):
        super().__init__()

        freq = (theta / 220.0) * 700 * (
            torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                    (dims // head) // 2, device=device, dtype=dtype) / 2595) - 1) / 1000

        position = torch.arange(0, ctx, device=device, dtype=dtype)
        sinusoid_inp = torch.einsum("q,k->qk", position, freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        n.register_buffer('emb', emb)

    def forward(n, x):
        return n.emb[None, :x.shape[1], :].to(x)

class PositionalR(nn.Module):
    def __init__(n, dims: int, ctx: int):
        super().__init__()
        n.dims = dims
        n.ctx = ctx
        n.pe = n.get_positional_encoding(ctx)

    def get_positional_encoding(n, ctx):
        pe = torch.zeros(ctx, n.dims)
        position = torch.arange(0, ctx, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n.dims, 2, dtype=torch.float32) * (-math.log(10000.0) / n.dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)

    def forward(n, x):
        ctx = x.size(1)
        pe = n.pe[:, :ctx, :]
        x = x * math.sqrt(n.dims)
        x = x + pe
        return x

def pitch_bias(f0):
    f0_flat = f0.squeeze().float()
    f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
    f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                f0_norm.unsqueeze(1)))
    return f0_sim.unsqueeze(0).unsqueeze(0)

def theta_freqs(dims, head, theta):
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
    freq = (theta / 220.0) * 700 * (
        torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), 
                (dims // head) // 2, device=theta.device, dtype=theta.dtype) / 2595) - 1) / 1000
    return freq

def apply_radii(freqs, x, ctx):
    F = x.shape[0] / ctx
    idx = torch.arange(ctx, device=device)
    idx = (idx * F).long().clamp(0, x.shape[0] - 1)
    x = x[idx]
    return torch.polar(x.unsqueeze(-1), freqs)

class frequencies(nn.Module):
    def __init__(n, dims, head, theta = 10000):
        super().__init__()
        dim = dims // head
        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        n.register_buffer('inv_freq', inv_freq)

    def forward(n, x):
        if isinstance(x, int):
            ctx = x 
        elif isinstance(x, torch.Tensor):    
            if x.dim()   == 1:
                ctx = x.shape            
            elif x.dim() == 2:
                batch, ctx = x.shape
            elif x.dim() == 3:
                batch, ctx, dims = x.shape
            else:
                batch, head, ctx, dims = x.shape
        t = torch.arange(ctx, device=device).type_as(n.inv_freq) # ctx = torch.arange(0, x.shape[1], device=device).unsqueeze(0)
        freqs = torch.einsum('q , k -> q k', t, n.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rrotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def sapply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rrotate_half(t) * pos.sin()

class axial_freqs(nn.Module):
    def __init__(n, dims, head, ctx, theta=10000, spec_shape=[]):

        time_frames, freq_bins = spec_shape
        time_frames = time_frames
        freq_bins = freq_bins
        
        time_theta = 50.0
        time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
        n.register_buffer('time_freqs', time_freqs)
        
        freq_theta = 100.0
        freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
        n.register_buffer('freq_freqs', freq_freqs)

        t = torch.arange(ctx, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, time_freqs)
        freqs_y = torch.outer(t_y, freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def compute_freqs_base(dim):
    mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 200 * mel_scale / 1000 

# class GELU(nn.Module):
#     def __init__(n):
#         super().__init__()
#     def forward(n, x):
#         return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)

def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()

def trace_x(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            print(f"  {func.__name__} returned shape: {result.shape}")
        return result
    return wrapper

def track_x(new_x, operation=""):  # track_x(x, "x") 
    x_id = [id(new_x)]
    if new_x is None:
        return new_x
    cur_id = id(new_x)
    if cur_id != x_id[0]:
        print(f"x FLOW: {x_id[0]} → {cur_id} in {operation}")
        x_id[0] = cur_id
    else:
        print(f"x REUSE: {cur_id} in {operation}")
    return new_x

def track_xa(new_xa, operation=""): # track_xa(xa, "xa - decoder")
    xa_id = [id(new_xa)] if new_xa is not None else [None]
    if new_xa is None:
        return new_xa
    cur_id = id(new_xa)
    if cur_id != xa_id[0]:
        print(f"xa FLOW: {xa_id[0]} → {cur_id} in {operation}")
        xa_id[0] = cur_id
    else:
        print(f"xa REUSE: {cur_id} in {operation}")
    return new_xa

def rotate_every_two(x):
    x = rearrange(x, '... (d k) -> ... d k', k = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d k -> ... (d k)')

def apply_rotory_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (k d) -> n k d', k = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n k)', k = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_emb(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_emb(q, k, cos, sin, ctx=None, unsqueeze_dim=1):
    cos = cos[0].unsqueeze(unsqueeze_dim)
    sin = sin[0].unsqueeze(unsqueeze_dim)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_emb(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_emb(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

class RotaryEmb(nn.Module):
    def __init__(n, dims, head):
        super().__init__()
        n.hd=dims // head
 
        n.scaling = 1.0
        n.theta = nn.Parameter((torch.tensor(360000, device=device, dtype=dtype)), requires_grad=False)  
        n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)
        n.freqs = 1.0 / (n.theta ** (torch.arange(0, dims // head, 2, device=device, dtype=dtype)[:(dims // head // 2)].float() / dims // head))

    def _compute_freqs_base(n):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.hd // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(n, x, ctx):
        # freqs = (n.theta / 220.0) * n.freqs_base
        freqs = n.freqs[None, :, None].float().expand(ctx.shape[0], -1, 1).to(x.device)
        position = ctx[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (freqs.float() @ position.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * n.scaling
            sin = emb.sin() * n.scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class positions(nn.Module): 
    def __init__(n, dims, head):
        super().__init__()

        # n.token = nn.Embedding(tokens, dims)
        n.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)        
        # n.pos = nn.Parameter(torch.empty(ctx, dims), requires_grad=True)
        n.text = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)  
        n.rotary = RotaryEmb(dims, head)
        # n.positional = PositionalR(ctx=ctx, dims=dims)

    def forward(n, x: Tensor, xa = None, kv_cache=None, fa=36000.0, fb=36000.0, freqs=False):
        # x = n.positional(x)
        # xa = n.positional(xa)
        x = x + n.text(x.shape[1], x.shape[-1], fa).to(device, dtype)   
        xa = xa + n.audio(xa.shape[1], xa.shape[-1], fb).to(device, dtype)    
        if freqs:
            ctx = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
            xf  = n.rotary(x, ctx)      
            ctx = torch.arange(0, xa.shape[1], device=xa.device).unsqueeze(0)
            xaf = n.rotary(xa, ctx)
            return xf, xaf
        else:
            return x, xa

class SLSTM(nn.Module):
    def __init__(n, dimension: int, layer: int = 2, skip: bool = False, bias = True, batch_first = True):
        super().__init__()
        n.skip = skip
        n.lstm = nn.LSTM(dimension, dimension, layer, bias, batch_first)

    def forward(n, x):
        x = x.permute(2, 0, 1)
        y, _ = n.lstm(x)
        if n.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y

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

def scaled_relu(x, sequence_length):
    relu_output = torch.relu(x)
    return relu_output / sequence_length

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for q in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
        tapprox += x**q / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

def taylor_masked(x, mask, order=2):
    tapprox = torch.zeros_like(x)
    unmasked = x.masked_select(mask) 
    approx_values = 1.0 + unmasked
    for q in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
        approx_values += unmasked**q / factorial_i
    tapprox.masked_scatter_(mask, approx_values)
    sum_approx = torch.sum(tapprox, dim=-1, keepdim=True)
    toutput = tapprox / (sum_approx + 1e-9) 
    toutput = toutput * mask 
    return toutput

def taylor_softmax2(x, mask=None, order=2):
    if mask is None:
        tapprox = 1.0 + x
        for q in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
            tapprox += x**q / factorial_i
        return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

    else:
        tapprox = torch.zeros_like(x)
        unmasked = x.masked_select(mask)
        tapprox = 1.0 + unmasked
        for q in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
            tapprox += unmasked**q / factorial_i

        tapprox_full = torch.zeros_like(x)
        tapprox_full.masked_scatter_(mask, tapprox)

        sum_approx = torch.sum(tapprox_full, dim=-1, keepdim=True)
        toutput = tapprox_full / (sum_approx + 1e-9)

        toutput = toutput * mask.float()
        return toutput

def taylor_softmax_2nd_order(x):
    exp_approx = 1 + x + (x**2) / 2
    return exp_approx / torch.sum(exp_approx, dim=-1, keepdim=True)

def taylor_softmax_approximation(x, order=2):
    if order == 0:
        return torch.ones_like(x) / x.size(-1) 
    elif order == 1:
        numerator = 1 + x
    elif order == 2:
        numerator = 1 + x + 0.5 * x**2
    else:
        raise NotImplementedError("Higher orders are not implemented yet.")
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator

def taylor_sine(x, order=5):
    result = torch.zeros_like(x)
    for q in range(order + 1):
        if q % 2 == 1:  
            term = x**q / torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
            if (q // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

def taylor_cosine(x, order=5):
    result = torch.zeros_like(x)
    for q in range(order + 1):
        if q % 2 == 0:  
            term = x**q / torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
            if (q // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

def vectorized_taylor_sine(x, order=5):
    og_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(1, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(og_shape)

def vectorized_taylor_cosine(x, order=5):
    og_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(0, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(og_shape)

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for q in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(q + 1, dtype=torch.float32)))
        tapprox += x**q / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

# def second_taylor(x: torch.Tensor, remove_even_power_dups: bool = False):

#     batch = x.shape[0]

#     # exp(qk) = 1 + qk + (qk)^2 / 2
#     x0 = x.new_ones((batch,)) 
#     x1 = x
#     x2 = einsum('... q, ... k -> ... q k', x, x) * (0.5 ** 0.5)

#     if remove_even_power_dups:
#         x2_diagonal = torch.diagonal(x2, dim1 = -2, dim2 = -1)  # Shape (batch_size, dims)
        
#         # Create a mask for the upper triangle (excluding the diagonal)
#         mask = torch.ones(x2.shape[-2:], dtype = torch.bool, device=x.device).triu(1)
        
#         # Select upper triangle elements and apply scaling
#         x2_upper_triangle = x2[:, mask] * (2 ** 0.5) # Using 2**0.5 for sqrt(2)
        
#         # Concatenate diagonal and upper triangle elements
#         x2 = torch.cat((x2_diagonal, x2_upper_triangle), dim = -1)
#     # else:
#     #     # If not removing duplicates, flatten the x2 matrix
#     #     x2 = x2.reshape(batch,)

#     # Concatenate the terms along the feature dimension
#     out = torch.cat((x0, x1, x2), dim = -1)
#     return out

# Example usage (assuming x is a Tensor)
# x_input = torch.randn(4, 3) # Example batch size 4, feature dim 3
# output_tensor = second_taylor_expansion_alt(x_input, remove_even_power_dups=True)
# print(output_tensor.shape)

def second_taylor(x: Tensor,  remove_even_power_dups = False):
    dtype, device, dim = x.dtype, x.device, x.shape[-1]
    x, ps = pack([x], '* d')
    lead_dims = x.shape[0]
    # exp(qk) = 1 + qk + (qk)^2 / 2
    x0 = x.new_ones((lead_dims,))
    x1 = x
    x2 = einsum('... q, ... k -> ... q k', x, x) * (0.5 ** 0.5)
  
    if remove_even_power_dups:
        x2_diagonal = torch.diagonal(x2, dim1 = -2, dim2 = -1)
        mask = torch.ones(x2.shape[-2:], dtype = torch.bool).triu(1)
        x2_upper_triangle = x2[:, mask] * sqrt(2)
        x2 = torch.cat((x2_diagonal, x2_upper_triangle), dim = -1)
    out, _ = pack((x0, x1, x2), 'b *')
    out, = unpack(out, ps, '* d')
    return out

def second_taylor_expansion(x: Tensor,  remove_even_power_dups = False):
    dtype, device, dim = x.dtype, x.device, x.shape[-1]
    x, ps = pack([x], '* d')
    lead_dims = x.shape[0]
    # exp(qk) = 1 + qk + (qk)^2 / 2
    x0 = x.new_ones((lead_dims,))
    x1 = x
    x2 = einsum('... q, ... k -> ... q k', x, x) * (0.5 ** 0.5)
  
    if remove_even_power_dups:
        x2_diagonal = torch.diagonal(x2, dim1 = -2, dim2 = -1)
        mask = torch.ones(x2.shape[-2:], dtype = torch.bool).triu(1)
        x2_upper_triangle = x2[:, mask] * sqrt(2)
        x2 = torch.cat((x2_diagonal, x2_upper_triangle), dim = -1)
    out, _ = pack((x0, x1, x2), 'b *')
    out, = unpack(out, ps, '* d')
    return out

def hz_to_midi(hz): # Converts Hz to MIDI note number. Handles 0 Hz as unvoiced.
    if hz == 0:
        return 0  # Special value for unvoiced
    return 12 * np.log2(hz / 440) + 69

def midi_to_pitch_class(midi_note): # Converts MIDI note number to pitch class (0-11).
    if midi_note == 0:
        return -1 # Represents unvoiced
    return int(round(midi_note)) % 12

def pcp(pitch_hz, num_pitch_classes=12):  # Each frame is a vector representing the strength of each pitch class. 
    pcp = torch.zeros(len(pitch_hz), num_pitch_classes)
    for q, hz in enumerate(pitch_hz):
        midi_note = hz_to_midi(hz)
        pitch_class = midi_to_pitch_class(midi_note)
        if pitch_class != -1: # If it's a voiced frame
            pcp[q, pitch_class] = 1 # Simple binary presence. Could be weighted by confidence.
    return pcp

def one_hot_pitch(pitch_hz, min_hz=50, max_hz=1000, num_bins=200): # Each bin represents a specific frequency range. 
    pitch_bins = np.linspace(min_hz, max_hz, num_bins + 1)
    one_hot = torch.zeros(len(pitch_hz), num_bins)
    for q, hz in enumerate(pitch_hz):
        if hz > 0:
            bin_idx = np.digitize(hz, pitch_bins) - 1  # Find the bin for the cur pitch
            if 0 <= bin_idx < num_bins:
                one_hot[q, bin_idx] = 1
    return one_hot

def gaussian_pitch(pitch_hz, min_hz=50, max_hz=1000, num_bins=200, sigma=1.0): # Pitch as a Gaussian distribution across frequency bins.
    pitch_bins_hz = np.linspace(min_hz, max_hz, num_bins)
    gaussian = torch.zeros(len(pitch_hz), num_bins)

    for q, hz in enumerate(pitch_hz):
        if hz > 0:
            midi_note = hz_to_midi(hz)                 # Calculate the bin index for the cur pitch
            midi_min = hz_to_midi(min_hz)                 # Map MIDI notes to the bin scale
            midi_max = hz_to_midi(max_hz)
            bin_idx_float = (midi_note - midi_min) / (midi_max - midi_min) * (num_bins - 1)

            for bin_j in range(num_bins):# Create a Gaussian distribution around the pitch
                bin_center_midi = midi_min + (bin_j / (num_bins - 1)) * (midi_max - midi_min) # Calculate the center of the bin in MIDI
                gaussian[q, bin_j] = torch.exp(-torch.tensor((midi_note - bin_center_midi)**2 / (2 * sigma**2)))
            gaussian[q] /= gaussian[q].sum() # Normalize each row to sum to 1 (optional, depends on your needs)
            
    return gaussian

def crepe_predict(audio, sample_rate, viterbi=False):
    import torchcrepe
    audio = audio.numpy().astype(np.float32)
    time, frequency, confidence, activation = torchcrepe.predict(
        audio, sample_rate=sample_rate, viterbi=viterbi)
    crepe_time = torch.from_numpy(time)
    crepe_frequency = torch.from_numpy(frequency)
    crepe_confidence = torch.from_numpy(confidence)
    crepe_activation = torch.from_numpy(activation)
    return crepe_time, crepe_frequency, crepe_confidence, crepe_activation

def rbf_scores(q, k, rbf_sigma=1.0, rbf_ratio=0.0):
    dot_scores = torch.matmul(q, k.transpose(-1, -2))
    if rbf_ratio <= 0.0:
        return dot_scores
    q_norm = q.pow(2).sum(dim=-1, keepdim=True)
    k_norm = k.pow(2).sum(dim=-1, keepdim=True)
    qk = torch.matmul(q, k.transpose(-1, -2))
    dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
    rbf_scores = torch.exp(-dist_sq / (2 * rbf_sigma**2))
    return (1 - rbf_ratio) * dot_scores + rbf_ratio * rbf_scores

def sliding_window_mask(q_len, k_len, window, device):
    idxs = torch.arange(q_len, device=device).unsqueeze(1)
    jdxs = torch.arange(k_len, device=device).unsqueeze(0)
    mask = (jdxs >= (idxs - window + 1)) & (jdxs <= idxs)
    return mask.float()

def mask_win(text_ctx, aud_ctx):
    mask = torch.tril(torch.ones(text_ctx, text_ctx, device=device, dtype=dtype), diagonal=0)
    audio_mask = torch.tril(torch.ones(text_ctx, aud_ctx - text_ctx, device=device, dtype=dtype))
    full_mask = torch.cat([mask, audio_mask], dim=-1)
    return full_mask

def maskc(ctx, device):
    return torch.tril(torch.ones(ctx, ctx, device=device, dtype=dtype), diagonal=0)

def attention_mask(ctx, batch_size=1, is_causal=True, padding_mask=None, expand=None, use_bool=False):
    if is_causal:
        if use_bool:
            mask = torch.ones((ctx, ctx), device = device, dtype = torch.bool).triu(1)
        else:
            mask = torch.empty(ctx, ctx, device=device, dtype=dtype).fill_(-np.inf).triu_(1)
        if expand:
            mask = mask.expand(batch_size, 1, ctx, ctx)

    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device, dtype=dtype)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).bool()
        mask = (mask.bool() | (~padding_mask)).float()
    return mask

def calculate_attention(q, k, v, mask=None, temp=1.0):
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    return out

def calculate_attentionb(q_norm, k_norm, v_iter, mask=None, temp=1.0):
    d_k = q_norm.size(-1)
    scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / (torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) / temp)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v_iter)
    return output

class LocalOut(nn.Module):
    def __init__(n, dims: int, head: int):
        super().__init__()
        n.head_dim = dims // head
        n.dims = dims
        n.q_module = nn.Linear(n.head_dim, n.head_dim)
        n.k_module = nn.Linear(n.head_dim, n.head_dim)
        n.v_module = nn.Linear(n.head_dim, n.head_dim)
        n.o_proj = nn.Linear(n.head_dim, n.head_dim)

    def _reshape_to_output(n, attn_output: Tensor) -> Tensor:
        batch, _, ctx, _ = attn_output.shape
        return attn_output.transpose(1, 2).contiguous().view(batch, ctx, n.dims)      

def qkv_init(dims, head):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims)
    lnb = nn.LayerNorm(dims)
    lnc = nn.LayerNorm(head_dim)
    lnd = nn.LayerNorm(head_dim)
    return q, k, v, o, lna, lnb, lnc, lnd

def shape(dims, head, q, k, v):
    batch_size = q.shape[0]
    ctx_q = q.shape[1]
    ctx_kv = k.shape[1]
    head_dim = dims // head

    q = q.view(batch_size, ctx_q, head, head_dim).transpose(1, 2)
    k = k.view(batch_size, ctx_kv, head, head_dim).transpose(1, 2)
    v = v.view(batch_size, ctx_kv, head, head_dim).transpose(1, 2)
    return q, k, v

def qkv(dims, head, q, k, v, x, xa):
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q(x) * scale
    k = k(xa) * scale
    v = v(xa)
    batch, ctx, dims = x.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

class KVCache(nn.Module):
    def __init__(n, max_batch_size, max_ctxgth, n_head, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_ctxgth, head_dim)
        n.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        n.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(n, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = n.k_cache
        v_out = n.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

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

class FEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, norm_type, norm=False):
        super().__init__()

        n.norm = norm
        n.act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        n.encoder = nn.Sequential(n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), n.act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   
        
        if n.norm:
            n.ln = get_norm(norm_type=norm_type, dims=dims) 
        else: 
            n.ln = None

    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        if n.norm:
            x = n.ln(x)
        return n.encoder(x).permute(0, 2, 1).contiguous().to(device=device, dtype=dtype)

# class FEncoder(nn.Module):
#     def __init__(n, mels, dims, head, act, norm_type, norm=False):
#         super().__init__()
        
#         n.head = head
#         n.head_dim = dims // head  
#         n.dims = dims
#         n.mels = mels
#         n.norm = norm
#         n.act_fn = get_activation(act)

#         n.encoder = nn.Sequential(
#            nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1), n.act_fn,
#            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), n.act_fn,
#            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)
#         if n.norm:
#             n.ln = get_norm(norm_type=norm_type, dims=dims) 
#         else: 
#             n.ln = None
        
#     def forward(n, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(0)        
#         x = n.encoder(x).permute(0, 2, 1).contiguous().to(device=device, dtype=dtype)
#         if n.norm:
#             x = n.ln(x)
#         return x

class WEncoder(nn.Module):
    def __init__(n, input_dims, dims, head, act="relu", downsample=True, target_length=None):
        super().__init__()
        
        n.head = head
        n.head_dim = dims // head
        n.dims = dims
        act_fn = get_activation(act)
        n.target_length = target_length

        if downsample:
            n.encoder = nn.Sequential(
                nn.Conv1d(1, dims, kernel_size=127, stride=64, bias=False),
                nn.Conv1d(dims, 2 * dims, kernel_size=7, stride=3),
                nn.Conv1d(2 * dims, dims, kernel_size=3, stride=2),
                nn.GroupNorm(num_groups=1, num_channels=dims, eps=1e-5))
        else:
            n.encoder = nn.Sequential(
               nn.Conv1d(input_dims, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
               nn.Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
               nn.Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)
                
    def _get_length(n, input_lengths: torch.LongTensor):
        conv1_length = int((input_lengths - 127) / 64 + 1)
        conv2_length = int((conv1_length - 7) / 3 + 1)
        conv3_length = int((conv2_length - 3) / 2 + 1)
        return conv3_length
        
    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = n.encoder(x).permute(0, 2, 1).contiguous()
        if n.target_length and x.shape[1] != n.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), n.target_length).transpose(1, 2)
        return x

class PEncoder(nn.Module):
    def __init__(n, input_dims, dims, head, act="relu", attend_pitch=False):
        super().__init__()
        
        n.head = head
        n.head_dim = dims // head
        n.dims = dims
   
        act_fn = get_activation(act)
        n.attend_pitch = attend_pitch

        if n.attend_pitch:
            n.q, n.k, n.v, n.o, n.scale = qkv_init(dims, head)
            n.mlp = nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.Linear(dims, dims),
            )
        else:
            n.q, n.k, n.v, n.o, n.scale = None, None, None, None, None
            n.mlp = None

        n.pitch_encoder = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)
        
    def rope_to(n, x):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, n.head, n.head_dim).permute(0, 2, 1, 3)
        freqs = n.rope(ctx)
        x = n.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x.squeeze(0)
        
    def forward(n, x, xa=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = n.pitch_encoder(x).permute(0, 2, 1).contiguous()
    
        if n.mlp is not None:
            x = n.mlp(x)

        if n.attend_pitch:
            if xa is not None:
                q, k, v = qkv(n.q, n.k, n.v, x=xa, xa=x, head=n.head)
                out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
                x = x + out

        return x

class feature_encoder(nn.Module):
    def __init__(n, mels, input_dims, dims, head, layer, act, features, feature=None, use_rope=False, spec_shape=None, debug=[], attend=False, target_length=None):
        """
        Feature encoder for audio processing.
        """
        super().__init__()

        n.dims = dims
        n.head = head
        n.head_dim = dims // head  
        n.dropout = 0.01 
        n.use_rope = use_rope
        n.attend = attend
        n.target_length = target_length
        n.feature = feature

        n.debug = debug
        act_fn = get_activation(act)

        if n.attend:
            n.mlp = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))
        else:
            n.q, n.k, n.v, n.o, n.scale = None, None, None, None, None
            n.mlp = None

        n.spectrogram = nn.Sequential(
           nn.Conv1d(mels, dims, kernel_size=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, groups=dims), act_fn)

        n.waveform = nn.Sequential(
           nn.Conv1d(1, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
           nn.Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
           nn.Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)

        n.pitch = nn.Sequential(
           nn.Conv1d(1, dims, kernel_size=7, stride=1, padding=3), act_fn,
           nn.Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        n.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        n.norm = RMSNorm(dims)

    def rope(n, x, xa=None, mask=None, feats=None, feature=None, layer=None):
        if isinstance(x, int):
            ctx = x 
        elif isinstance(x, torch.Tensor):
            ctx = x.shape[1] if x.dim() > 1 else x.shape[0]
            batch, ctx, dims = x.shape[0], ctx, x.shape[-1]

            x = x.view(batch, ctx, n.head, n.head_dim).permute(0, 2, 1, 3)
        freqs = n.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = n.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x

    def mel_scalar(n, freq: float) -> float:
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def forward(n, x, xa=None, mask=None, feats=None, feature=None, layer=None, max_tscale=36000):
        target_length = x.shape[1] if n.target_length is None else n.target_length

        if feature == "pitch":
            xp = x.clone()
            enc_dict = feats if feats is not None else {}
            enc_dict = dict(enc_dict)  
            enc_dict["f0"] = xp
            feats = enc_dict
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.pitch(x).permute(0, 2, 1)
  
        if feature == "phase":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.pitch(x).permute(0, 2, 1)

        if feature == "waveform":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.waveform(x).permute(0, 2, 1)
            if target_length and x.shape[1] != n.target_length:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)
        
        if feature == "harmonics":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.spectrogram(x).permute(0, 2, 1)

        if feature == "aperiodic":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.spectrogram(x).permute(0, 2, 1)            

        if feature == "spectrogram":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = n.spectrogram(x).permute(0, 2, 1)

        if n.use_rope:
            x = x + n.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
            x = n.rope(x=x, xa=None, mask=None, feats=feats, feature=feature, layer=layer)
        else:
            max_tscale = x.shape[1] * 1000 if max_tscale is None else max_tscale
            x = x + n.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
        x = nn.functional.dropout(x, p=n.dropout, training=n.training)
        x = n.norm(x)

        x = nn.functional.dropout(x, p=n.dropout, training=n.training)
        x = n.norm(x)
        return x

# class OneShot(nn.Module):
#     def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
#         super().__init__()
#         if features is None:    
#             features = ["spectrogram", "waveform", "pitch", "aperiodic", "harmonics"]
#         n.head = head
#         n.head_dim = dims // head
#         n.scale = 1.0 // len(features) if features else scale

#         n.q = Linear(dims, dims)
#         n.k = Linear(dims, dims)

#     def forward(n, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
#         B, L, D = x.shape
#         K = xa.size(1)
#         q = n.q(x).view(B, L, n.head, n.head_dim).transpose(1,2)
#         k = n.k(xa).view(B, K, n.head, n.head_dim).transpose(1,2)
#         bias = (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.head_dim)

#         return bias

# class curiosity(nn.Module):
#     def __init__(n, d, h, bias=True):
#         super().__init__()
#         n.h  = h
#         n.dh = d // h
#         n.qkv = nn.Linear(d, d * 3, bias=bias)
#         n.qkv_aux = nn.Linear(d, d * 3, bias=bias)
#         n.o  = nn.Linear(d, d, bias=bias)
#         n.g  = nn.Parameter(torch.zeros(h))

#     def split(n, x):
#         b, t, _ = x.shape
#         return x.view(b, t, n.h, n.dh).transpose(1, 2)

#     def merge(n, x):
#         b, h, t, dh = x.shape
#         return x.transpose(1, 2).contiguous().view(b, t, h * dh)

#     def forward(n, x, xa, mask=None):

#         q, k, v   = n.qkv(x).chunk(3, -1)
#         qa, ka, va = n.qkv_aux(xa).chunk(3, -1)
#         q, k, v   = map(n.split, (q, k, v))
#         qa, ka, va = map(n.split, (qa, ka, va))
#         dots      = (q @ k.transpose(-2, -1)) / n.dh**0.5
#         dots_aux  = (q @ ka.transpose(-2, -1)) / n.dh**0.5
#         if mask is not None: dots = dots.masked_fill(mask, -9e15)
#         p   = dots.softmax(-1)
#         pa  = dots_aux.softmax(-1)
#         h_main = p  @ v
#         h_aux  = pa @ va
#         g = torch.sigmoid(n.g).view(1, -1, 1, 1)
#         out = n.merge(h_main * (1 - g) + h_aux * g)
#         return n.o(out)

class Conv2d(nn.Conv2d):
    def _conv_forward(
        n, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Linear(nn.Module):
    def __init__(n, in_dims: int, out_dims: int, bias: bool = True) -> None:
        super(Linear, n).__init__()
        n.linear = nn.Linear(in_dims, out_dims, bias=bias)
        init.xavier_uniform_(n.linear.weight)
        if bias:
            init.zeros_(n.linear.bias)
    def forward(n, x: Tensor) -> Tensor:
        return n.linear(x)
    
class RMSNorm(nn.Module):
    def __init__(n, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(RMSNorm, n).__init__()
        if isinstance(dims, int):
            n.normalized_shape = (dims,)
        else:
            n.normalized_shape = tuple(dims)
        n.eps = eps
        n.elementwise_affine = elementwise_affine
        if n.elementwise_affine:
            n.weight = nn.Parameter(torch.empty(n.normalized_shape))
            init.ones_(n.weight)  
        else:
            n.register_parameter("weight", None)
    def forward(n, x):
        return F.rms_norm(x, n.normalized_shape, n.weight, n.eps)
    
# def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
#                weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
#                eps: float = 1e-5) -> Tensor:
#     return F.layer_norm(x, normalized_shape, weight, bias, eps)

class SelfCriticalRL(nn.Module):
    def __init__(n, model, tokenizer, reward_fn):
        super().__init__()
        n.model = model
        n.tokenizer = tokenizer
        n.reward_fn = reward_fn

    def forward(n, input_ids, features, labels=None, max_len=128, feature_name="spectrogram"):

        with torch.no_grad():
            greedy_ids = n.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len)
        greedy_text = [n.tokenizer.decode(ids) for ids in greedy_ids]
        sampled_ids = n.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len, do_sample=True, top_k=5)
        sampled_text = [n.tokenizer.decode(ids) for ids in sampled_ids]
        
        rewards = []
        baseline = []
        for s, g, ref in zip(sampled_text, greedy_text, labels):
            ref_text = n.tokenizer.decode(ref)
            rewards.append(n.reward_fn(s, ref_text))
            baseline.append(n.reward_fn(g, ref_text))
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        baseline = torch.tensor(baseline, device=device, dtype=torch.float)
        advantage = rewards - baseline
        logits = n.model(input_ids=sampled_ids, **{feature_name: features})["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_seq = torch.gather(log_probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_sum = log_probs_seq.sum(dim=1)
        loss = -(advantage * log_probs_sum).mean()
        return loss

class SelfTrainingModule(nn.Module):
    def __init__(n, model, tokenizer, quality_fn=None, threshold=0.8):
        super().__init__()
        n.model = model
        n.tokenizer = tokenizer
        n.quality_fn = quality_fn
        n.threshold = threshold

    def generate_pseudo_labels(n, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        with torch.no_grad():
            pred_ids = n.model.generate(input_ids=unlabeled_batch, **{feature_name: features}, max_length=max_len)

        if n.quality_fn is not None:
            quality_scores = n.quality_fn(pred_ids, n.model, features)
            mask = quality_scores > n.threshold
            pred_ids = pred_ids[mask]
        return pred_ids

    def forward(n, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        pseudo_labels = n.generate_pseudo_labels(unlabeled_batch, features, max_len, feature_name=feature_name)
        logits = n.model(input_ids=unlabeled_batch, **{feature_name: features}, labels=pseudo_labels)["logits"]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), pseudo_labels.view(-1), ignore_index=0)
        return loss

def confidence_indicator(pred_ids, model, features):
    with torch.no_grad():
        logits = model(input_ids=pred_ids, **features)["logits"]
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return max_probs.mean(dim=1)

def wer_reward(hyp, ref):
    hyp_words = hyp.split()
  
    ref_words = ref.split()
    d = [[0] * (len(ref_words)+1) for _ in range(len(hyp_words)+1)]
    for q in range(len(hyp_words)+1):
        d[q][0] = q
    for k in range(len(ref_words)+1):
        d[0][k] = k
    for q in range(1, len(hyp_words)+1):
        for k in range(1, len(ref_words)+1):
            if hyp_words[q-1] == ref_words[k-1]:
                d[q][k] = d[q-1][k-1]
            else:
                d[q][k] = 1 + min(d[q-1][k], d[q][k-1], d[q-1][k-1])
    wer = d[-1][-1] / max(1, len(ref_words))
    return -wer

def clean_ids(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(id) for id in ids if id != -100 and id != pad_token_id and id != bos_token_id and id != eos_token_id]

def clean_batch(batch_ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    return [clean_ids(seq, pad_token_id, bos_token_id, eos_token_id) for seq in batch_ids]

def setup_tokenizer(dir: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{dir}")
    orig_encode = tokenizer.encode
    orig_decode = tokenizer.decode

    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids

    def bdec(ids_list, pad_token_id=0, bos_token_id=1, eos_token_id=2, skip_special_tokens=True):
        results = []
        if isinstance(ids_list, torch.Tensor):
            ids_list = ids_list.tolist()
        elif isinstance(ids_list, np.ndarray):
            ids_list = ids_list.tolist()
        for ids in ids_list:
            ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
            results.append(orig_decode(ids))
        return results

    def dec(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        ids = [int(id) for id in ids if id not in (pad_token_id, bos_token_id, eos_token_id, -100)]
        return orig_decode(ids)

    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")

    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.decode = dec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

def tokenize_feature(audio, labels):
    if isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            ctx = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            ctx = audio.unsqueeze(1)
        elif audio.dim() == 3:
            ctx = audio
    target_length = len(labels)
    current_length = ctx.shape[-1]
    if current_length > target_length:
        tokens = F.adaptive_avg_pool1d(ctx, target_length)
    else:
        tokens = F.interpolate(ctx, size=target_length, mode='linear', align_corners=False)
    return tokens

def load_wave(wave_data, sample_rate=16000):
    if isinstance(wave_data, str):
        waveform, sample_rate = torchaudio.load(uri=wave_data, normalize=True, backend="ffmpeg")
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sample_rate = wave_data["sampling_rate"]
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

def spectrogram(audio, sample_rate=16000, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
    torch_windows = {
        'hann': torch.hann_window,
        'hamming': torch.hamming_window,
        'blackman': torch.blackman_window,
        'bartlett': torch.bartlett_window,
        'ones': torch.ones,
        None: torch.ones,
    }

    if isinstance(window_fn, str):
        window_fn = torch_windows[window_fn]
    if window_fn is None:
        window_fn = torch.ones(n_fft)
    if isinstance(window_fn, torch.Tensor):
        window_fn = window_fn.to(device)
    return torchaudio.functional.spectrogram(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window_fn, center=True, pad_mode="reflect", power=1.0)

def exact_div(x, y):
    assert x % y == 0
    return x // y

def load_audio(file: str, sr: int = 16000):

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-q", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = 30, *, axis: int = -1):  # N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk 
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = "D:/newmodel/mod6/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    n_fft = 400,
    hop_length = 160,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    spectrogram_tensor = (log_spec + 4.0) / 4.0
    return spectrogram_tensor

def audio_token(audio, labels, sample_rate=16000, hop_length=256, strides=1):

    frames_per_second = exact_div(sample_rate, hop_length)  
    # key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
    tokens_per_second = exact_div(sample_rate,  hop_length * strides) # audio "tokens"  20ms per audio token (exampe)
    return tokens_per_second

def harmonics_and_aperiodics(audio, f0, t, sample_rate):
    import pyworld as pw
    wav_np = audio.numpy().astype(np.float64)
    sp = pw.cheaptrick(wav_np, f0, t, sample_rate, fft_size=256)
    ap = pw.d4c(wav_np, f0, t, sample_rate, fft_size=256)
    harmonic_tensor = torch.from_numpy(sp)
    aperiodic_tensor = torch.from_numpy(ap)
    harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
    aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
    harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
    aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
    return harmonic_tensor, aperiodic_tensor

def mfcc(audio, sample_rate, n_mels, n_fft, hop_length, window_fn=torch.hann_window):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mels,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window_fn": window_fn,
            "n_mels": n_mels,
            "center": True,
            "pad_mode": "reflect",
            "norm": None,
            "mel_scale": "htk",
        }
    )
    mfcc_tensor = transform(audio)
    return mfcc_tensor

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def max_neg(tensor):
    return -torch.finfo(tensor.dtype).max

def pitch_tokens(audio, labels, sample_rate=16000, hop_length=160, mode="mean", audio_bos=None):
    f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
    f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
    duration = len(audio) / sample_rate
    T = len(labels)
    tok_dur = duration / T
    starts = torch.arange(T) * tok_dur
    ends = starts + tok_dur
    start = torch.searchsorted(torch.from_numpy(t), starts, side="left")
    end = torch.searchsorted(torch.from_numpy(t), ends, side="right")
    ptok = torch.zeros(T, dtype=torch.float32)
    for q in range(T):
        lo, hi = start[q], max(start[q]+1, end[q])
        seg = torch.from_numpy(f0)[lo:hi]
        if mode == "mean":
            ptok[q] = seg.mean()
        elif mode == "median":
            ptok[q] = torch.median(seg)
        else:
            ptok[q] = seg[-1]
    ptok[ptok < 100.0] = 0.0
    bos_token = audio_bos if audio_bos is not None else (ptok[0] if len(ptok) > 0 else 0.0)
    tensor = torch.cat([torch.tensor([bos_token]), ptok])
    return torch.where(tensor == 0.0, torch.zeros_like(tensor), (tensor - 71.0) / (500.0 - 71.0))

class DynamicLayerNorm(nn.Module):
    def __init__(n, normalized_shape, summary_dim):
        super(DynamicLayerNorm, n).__init__()
        n.normalized_shape = normalized_shape
        n.gamma_generator = nn.Linear(summary_dim, normalized_shape)
        n.beta_generator = nn.Linear(summary_dim, normalized_shape)

    def forward(n, x, summary_vector):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        gamma = n.gamma_generator(summary_vector)
        beta = n.beta_generator(summary_vector)
        return gamma * x_norm + beta

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

# class SummaryVector(nn.Module):
#     def __init__(n, input_dim, dims, layer, summary_dim, output_dim):
#         super(SummaryVector, n).__init__()
#         n.lstm = nn.LSTM(input_dim, dims, layer, batch_first=True)
#         n.dln = DynamicLayerNorm(dims, summary_dim) 
#         n.fc = nn.Linear(dims, output_dim)
#         n.summary_lstm = nn.LSTM(input_dim, summary_dim, 1, batch_first=True) # A single layer LSTM for summary

#     def forward(n, x):
#         _, (summary_vector, _) = n.summary_lstm(x)
#         summary_vector = summary_vector.squeeze(0) # Remove the layer dimension
#         lstm_output, _ = n.lstm(x) 
#         normalized_lstm_output = n.dln(lstm_output, summary_vector)
#         output = n.fc(normalized_lstm_output)
#         return output

class DynamicLayerNorm(nn.LayerNorm):
    def __init__(n, normalized_shape, summary_dim):
        super(DynamicLayerNorm, n).__init__()
        n.normalized_shape = normalized_shape
        n.gamma_generator = nn.Linear(summary_dim, normalized_shape)
        n.beta_generator = nn.Linear(summary_dim, normalized_shape)

    def forward(n, x, summary_vector):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        gamma = n.gamma_generator(summary_vector)
        beta = n.beta_generator(summary_vector)
        gamma = gamma.unsqueeze(1) # shape becomes (batch, 1, normalized_shape)
        beta = beta.unsqueeze(1)   # shape becomes (batch, 1, normalized_shape)
        return gamma * x_norm + beta

class SummaryVector(nn.Module):
    def __init__(n, input_dim, dims, layer, summary_dim, output_dim):
        super(SummaryVector, n).__init__()
        n.lstm = nn.LSTM(input_dim, dims, layer, batch_first=True)
        n.dln = DynamicLayerNorm(dims, summary_dim) 
        n.fc = nn.Linear(dims, output_dim)
        n.summary_lstm = nn.LSTM(input_dim, summary_dim, 1, batch_first=True) # A single layer LSTM for summary

    def forward(n, x):
        _, (summary_vector, _) = n.summary_lstm(x)
        summary_vector = summary_vector.squeeze(0) # Remove the layer dimension
        lstm_output, _ = n.lstm(x) 
        normalized_lstm_output = n.dln(lstm_output, summary_vector)
        output = n.fc(normalized_lstm_output)
        return output

class SummaryVector(nn.Module):  # Simple
    def __init__(n, input_dim, dims, layer, summary_dim, output_dim):
        super(SummaryVector, n).__init__()
        n.lstm = nn.LSTM(input_dim, dims, layer, batch_first=True)

        n.dln = DynamicLayerNorm(dims, summary_dim) 
        n.fc = nn.Linear(dims, output_dim)
        n.summary_extractor = nn.Linear(input_dim, summary_dim)

    def forward(n, x):
        summary_vector = n.summary_extractor(x.mean(dim=1)) 
        lstm_output, _ = n.lstm(x) 
        normalized_lstm_output = n.dln(lstm_output, summary_vector)
        output = n.fc(normalized_lstm_output)
        return output

class AttentionPooling(nn.Module):
    def __init__(n, encoder_hidden_dim):
        super(AttentionPooling, n).__init__()
        n.attention_query_generator = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        n.attention_mechanism = nn.Linear(encoder_hidden_dim, 1)

    def forward(n, encoder_outputs):
        query = torch.mean(encoder_outputs, dim=1) # shape: (batch_size, encoder_hidden_dim)
        scores = n.attention_mechanism(torch.tanh(encoder_outputs + n.attention_query_generator(query).unsqueeze(1)))
        attention_weights = F.softmax(scores, dim=1) # shape: (batch_size, sequence_length, 1)
        pooled_output = torch.sum(attention_weights * encoder_outputs, dim=1)
        return pooled_output, attention_weights  # pooled_output shape: (batch_size, encoder_hidden_dim)

class SummaryVector_Attention(nn.Module):
    def __init__(n, input_dim, dims, layer, summary_dim, output_dim):
        super(SummaryVector_Attention, n).__init__()
        n.lstm = nn.LSTM(input_dim, dims, layer, batch_first=True)
        n.attention_pooling = AttentionPooling(dims) # AttentionPooling layer
        n.dln = DynamicLayerNorm(dims, summary_dim) 
        n.fc = nn.Linear(dims, output_dim)
        n.summary_extractor = nn.LSTM(input_dim, summary_dim, 1, batch_first=True) 

    def forward(n, x):
        _, (summary_vector, _) = n.summary_lstm(x) # using n.summary_lstm from previous example
        summary_vector = summary_vector.squeeze(0)
        lstm_output, _ = n.lstm(x) 
        summary_vector_from_attention, attention_weights_viz = n.attention_pooling(lstm_output)        
        normalized_lstm_output = n.dln(lstm_output, summary_vector_from_attention) # Now the summary is attention-based
        output = n.fc(normalized_lstm_output.mean(dim=1)) # Example: average pooling after DLN
        return output

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

def extract_features(batch, tokenizer, waveform=False, spec=False, pitch_tokens=False, pitch=False, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, phase_mod=False, crepe=False, aperiodics=False, dummy=False, mels=128, n_fft= 1024):

    sample_rate = batch["audio"]["sampling_rate"]

    # if dummy:
    #     labels = torch.ones(32, 1024)
    # else:
    labels = tokenizer.encode(batch["transcription"])
    audio = load_wave(batch["audio"], sample_rate)
    # tokens = tokenize_feature(audio, labels)
        
    # pitch_tensor_hz = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    # if pitch_tensor_hz.dim() > 1 and pitch_tensor_hz.shape[0] == 1:
    #     pitch_tensor_hz = pitch_tensor_hz.squeeze(0) 
    # ctx = pitch_tensor_hz.shape[0]
    # print(f"Original pitch tensor shape from torchaudio: {pitch_tensor_hz.shape}")
    # pitch_hz_np = pitch_tensor_hz.numpy()
    # num_pitch_classes = 128
    # pcp_data = pcp(pitch_hz_np, num_pitch_classes=num_pitch_classes)
    # pcp_data = pcp_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_classes, ctx)
    # print(f"PCP data shape: {pcp_data.shape}")

    # dims = 512
    #nn.Conv1d_pcp = nn.Conv1d(in_channels=num_pitch_classes, out_channels=dims, kernel_size=3, stride=1, padding=1)
    # output_pcp =nn.Conv1d_pcp(pcp_data)
    # print(f"Conv1D output shape for PCP: {output_pcp.shape}")

    # =========================================================================
    # 2. One-Hot Encoding or Gaussian Distribution
    # =========================================================================
    # min_hz, max_hz, num_pitch_bins = 50, 1000, 200 # Example parameters
    # one_hot_data = one_hot_pitch(pitch_hz_np, min_hz=min_hz, max_hz=max_hz, num_bins=num_pitch_bins)
    # one_hot_data = one_hot_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_bins, ctx)
    # print(f"One-Hot pitch data shape: {one_hot_data.shape}")

    #nn.Conv1d_one_hot = nn.Conv1d(in_channels=num_pitch_bins, out_channels=dims, kernel_size=3, stride=1, padding=1)
    # output_one_hot =nn.Conv1d_one_hot(one_hot_data)
    # print(f"Conv1D output shape for One-Hot pitch: {output_one_hot.shape}")

    # gaussian_data = gaussian_pitch(pitch_hz_np, min_hz=min_hz, max_hz=max_hz, num_bins=num_pitch_bins, sigma=0.5)
    # gaussian_data = gaussian_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_bins, ctx)
    # print(f"Gaussian pitch data shape: {gaussian_data.shape}")

    #nn.Conv1d_gaussian = nn.Conv1d(in_channels=num_pitch_bins, out_channels=dims, kernel_size=3, stride=1, padding=1)
    # output_gaussian =nn.Conv1d_gaussian(gaussian_data)
    # print(f"Conv1D output shape for Gaussian pitch: {output_gaussian.shape}")

    # =========================================================================
    # 3. Learned Feature Extraction (nn.Embedding)
    # =========================================================================

    # min_midi = 21 # A0
    # max_midi = 108 # C8
    # num_pitch_categories = max_midi - min_midi + 1 + 1 # +1 for unvoiced at index 0

    # quantized_pitch_sequence = torch.zeros(ctx, dtype=torch.long)
    # for q, hz in enumerate(pitch_hz_np):
    #     if hz > 0:
    #         midi_note = hz_to_midi(hz)
    #         quantized_midi = int(round(np.clip(midi_note, min_midi, max_midi)))
    #         quantized_pitch_sequence[q] = quantized_midi - min_midi + 1

    # embedding_layer = nn.Embedding(num_embeddings=num_pitch_categories, dims=dims)
    # embedded_pitch = embedding_layer(quantized_pitch_sequence)
    # embedded_pitch = embedded_pitch.unsqueeze(0).transpose(1, 2) # (1, dims, ctx)
    # print(f"Embedded pitch data shape: {embedded_pitch.shape}")

    #nn.Conv1d_embedding = nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=3, stride=1, padding=1)
    # output_embedding =nn.Conv1d_embedding(embedded_pitch)
    # print(f"Conv1D output shape for embedded pitch: {output_embedding.shape}")
 
    if crepe:
        crepe_time, crepe_frequency, crepe_confidence, crepe_activation = crepe_predict(audio, sample_rate, viterbi=True)

    else:
        crepe_time = None
        crepe_frequency = None
        crepe_confidence = None
        crepe_activation = None

    # # hard-coded audio hyperparameters
    # SAMPLE_RATE = 16000
    # N_FFT = 400
    # HOP_LENGTH = 160
    # CHUNK_LENGTH = 30
    # N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    # N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

    # N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
  
    # FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
    # TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

            # spectrogram_config = {
            #     "hop_length": 256,
            #     "f_min": 150,
            #     "f_max": 2000,
            #     "n_mels": 128,
            #     "n_fft": 1024,
            #     "sample_rate": 16000,
            #     "pad_mode": "constant",
            #     "center": True, 
            #     "power": 1.0,
            #     "window_fn": torch.hann_window,
            #     "mel_scale": "htk",
            #     "norm": None,
            #     "normalized": False,
            # }

        # spectrogram_config = {
        #     "hop_length": 1280,  # Increased significantly for very low temporal resolution
        #     "f_min": 500,        # Narrower range, especially eliminating some lower frequencies
        #     "f_max": 1500,       # Narrower range, limiting higher frequencies
        #     "n_mels": 8,         # Significantly reduced for very low frequency resolution
        #     "n_fft": 128,        # Reduced for lower frequency resolution (related to hop_length and sample_rate)
        #     "sample_rate": 8000, # Also reduced to lower the effective frequency range
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # spectrogram_config = {
        #     "hop_length": 320,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,
        #     "n_fft": 512,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 1: Reduce temporal resolution (increase hop_length)
        # spectrogram_config_v1 = {
        #     "hop_length": 480,  # Increased hop_length
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 128,
        #     "n_fft": 1024,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 2: Reduce frequency resolution (decrease n_mels)
        # spectrogram_config_v2 = {
        #     "hop_length": 256,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,  # Decreased n_mels
        #     "n_fft": 1024,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 3: Reduce frequency resolution (decrease n_fft)
        # spectrogram_config_v3 = {
        #     "hop_length": 256,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 128,
        #     "n_fft": 512,  # Decreased n_fft
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # # Option 4: Combined reduction (example)
        # spectrogram_config = {
        #     "hop_length": 320,
        #     "f_min": 150,
        #     "f_max": 2000,
        #     "n_mels": 64,
        #     "n_fft": 512,
        #     "sample_rate": 16000,
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }

        # transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        # mel_spectrogram = transform(audio)
        # log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        # log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        # spectrogram_tensor = (log_mel + 4.0) / 4.0
        # return spectrogram_tensor

        # transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        # return transform(audio).log10()

    def mel_spectrogram(audio, sample_rate):

        spectrogram_config = {
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
        # spectrogram_config = {
        #     "hop_length": 1280,  # Increased significantly for very low temporal resolution
        #     "f_min": 500,        # Narrower range, especially eliminating some lower frequencies
        #     "f_max": 1500,       # Narrower range, limiting higher frequencies
        #     "n_mels": 8,         # Significantly reduced for very low frequency resolution
        #     "n_fft": 128,        # Reduced for lower frequency resolution (related to hop_length and sample_rate)
        #     "sample_rate": 8000, # Also reduced to lower the effective frequency range
        #     "pad_mode": "constant",
        #     "center": True,
        #     "power": 1.0,
        #     "window_fn": torch.hann_window,
        #     "mel_scale": "htk",
        #     "norm": None,
        #     "normalized": False,
        # }
        transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        mel_spectrogram = transform(audio)
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spectrogram_tensor = (log_mel + 4.0) / 4.0
        return spectrogram_tensor

    if spec: 
        if dummy:
            spectrogram_tensor = torch.ones(mels, 1024)
        else:
            spectrogram_tensor = mel_spectrogram(audio, sample_rate)
            # spectrogram_tensor = mel_spectrogram(audio=audio, n_mels=128, n_fft=400, hop_length=160, padding=0, device=device)
            # spectrogram_tensor = FEncode(spectrogram_tensor)
    else:
        spectrogram_tensor = None

    if pitch_tokens or harmonics or aperiodics:
        wavnp = audio.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

    if pitch_tokens:
        audio = torch.from_numpy(wavnp)
        t2 = torch.from_numpy(t)
        audio_duration = len(audio) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t2, token_starts, side="left")
        end_idx = torch.searchsorted(t2, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)
        for q in range(T):
            lo, hi = start_idx[q], max(start_idx[q]+1, end_idx[q])
            segment = f0_np[lo:hi]
            if mode == "mean":
                pitch_tok[q] = segment.mean()
            elif mode == "median":
                pitch_tok[q] = torch.median(segment)
            else:
                pitch_tok[q] = segment[-1]
        pitch_tok[pitch_tok < 100.0] = 0.0
        bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
        pitch_tokens_tensor = torch.cat([torch.tensor([bos_pitch]), pitch_tok])
        pitch_tokens_tensor = torch.where(pitch_tokens_tensor == 0.0, torch.zeros_like(pitch_tokens_tensor), (pitch_tokens_tensor - 71.0) / (500.0 - 71.0))
    else:
        pitch_tokens_tensor = None

    if phase_mod:
        tframe = torch.mean(t2[1:] - t2[:-1])
        phi0 = 0.0
        omega = 2 * torch.pi * f0_tensor
        dphi = omega * tframe
        phi = torch.cumsum(dphi, dim=0) + phi0
        phase = torch.remainder(phi, 2 * torch.pi)
    else:
        phase = None

    if pitch:
        if dummy:
            p_tensor = torch.ones(1, 1024)
        else:
            p_tensor = torchaudio.functional.detect_pitch_frequency(audio, sample_rate).unsqueeze(0)
        # p_tensor = PEncode(p_tensor)

        # pitch_tensor_hz = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
        # if pitch_tensor_hz.dim() > 1 and pitch_tensor_hz.shape[0] == 1:
        #     pitch_tensor_hz = pitch_tensor_hz.squeeze(0) 
        # ctx = pitch_tensor_hz.shape[0]
        # # print(f"Original pitch tensor shape from torchaudio: {pitch_tensor_hz.shape}")
        # pitch_hz_np = pitch_tensor_hz.numpy()
        # num_pitch_classes = 128
        # pcp_data = pcp(pitch_hz_np, num_pitch_classes=num_pitch_classes)
        # p_tensor = pcp_data.unsqueeze(0).transpose(1, 2) # (1, num_pitch_classes, ctx)
        # print(f"PCP data shape: {pcp_data.shape}")

    else:
        p_tensor = None

    if harmonics or aperiodics:
        spnp = pw.cheaptrick(wavnp, f0_np, t, sample_rate, fft_size=256)
        apnp = pw.d4c(wavnp, f0_np, t, sample_rate, fft_size=256)
        harmonic_tensor = torch.from_numpy(spnp)
        aperiodic_tensor = torch.from_numpy(apnp)
        harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
        aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
        harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
        aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
    else:
        harmonic_tensor = None
        aperiodic_tensor = None

    if waveform:

        if dummy:
            wave_tensor = torch.ones(1, 1024)
        else:
            wave_tensor = audio
        # wave_tensor = WEncode(wave_tensor)
    else:
        wave_tensor = None

    # if dummy:   
    #     batch_size = 1
    #     ctx = 1024
    #     if spectrogram_tensor is not None:
    #         # spectrogram_tensor = torch.randn(mels, ctx)
    #         spectrogram_tensor = torch.ones(mels, ctx)
        
    #     if p_tensor is not None:
    #         p_tensor = torch.ones_like(p_tensor) 
        
    #     if pitch_tokens_tensor is not None:
    #         dummy_tensor = torch.ones_like(pitch_tokens_tensor)
        
    #     else:
    #         batch_size = 128
    #         ctx = 1024
    #         dummy_tensor = torch.ones(batch_size, ctx)
    #         dummy_tensor = dummy_tensor.to(device)
    # else:
    #     dummy_tensor = None
        
    if debug:
        print(f"['pitch_tokens']: {pitch_tokens_tensor.shape if pitch_tokens else None}")
        print(f"['harmonic']: {harmonic_tensor.shape if harmonics else None}")
        print(f"['aperiodic']: {aperiodic_tensor.shape if aperiodics else None}")
        print(f"['spectrogram']: {spectrogram_tensor.shape if spec else None}")
        print(f"['waveform']: {wave_tensor.shape if waveform else None}")
        print(f"['labels']: {len(labels) if labels else None}")
        print(f"['phase']: {phase.shape if phase else None}")
        print(f"['pitch']: {p_tensor.shape if pitch else None}")
        print(f"['crepe_time']: {crepe_time.shape if crepe else None}")  
        print(f"['crepe_frequency']: {crepe_frequency.shape if crepe else None}")
        print(f"['crepe_confidence']: {crepe_confidence.shape if crepe else None}")
        print(f"['crepe_activation']: {crepe_activation.shape if crepe else None}")
        # print(f"['dummy']: {dummy_tensor.shape if dummy else None}")

    return {
        "waveform": wave_tensor if waveform else None,
        "spectrogram": spectrogram_tensor if spec else None,
        "pitch_tokens": pitch_tokens_tensor if pitch_tokens else None,
        "pitch": p_tensor if pitch else None,
        "harmonic": harmonic_tensor if harmonics else None,
        "aperiodic": aperiodic_tensor if aperiodics else None,  
        "labels": labels,
        "phase": phase if phase_mod else None,
        "crepe_time": crepe_time if crepe else None,
        "crepe_frequency": crepe_frequency if crepe else None,
        "crepe_confidence": crepe_confidence if crepe else None,
        "crepe_activation": crepe_activation if crepe else None,
        # "dummy": dummy_tensor if dummy else None,
    }

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    import librosa
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=None, max_ctx=2048):

    if extract_args is None:
        extract_args = {
        "waveform": False,
        "spec": False,
        "f0": False,
        "pitch_tokens": False,
        "pitch": False,
        "harmonic": False,
        "aperiodic": False,
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
        "phase_mod": False,
        "crepe": False,
        "dummy": False,
        }

    if load_saved:
        if cache_dir is None:
            cache_dir = "./processed_datasets"
        else:
            cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        cache_file_train = os.path.join(cache_dir, "train.arrow")
        cache_file_test = os.path.join(cache_dir, "test.arrow")

        if os.path.exists(cache_file_train) and os.path.exists(cache_file_test):
            from datasets import Dataset
            train_dataset = Dataset.load_from_disk(cache_file_train)
            test_dataset = Dataset.load_from_disk(cache_file_test)
            return train_dataset, test_dataset   

    if sanity_check:
        test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).cast_column("audio", Audio(sampling_rate=sample_rate)).take(1)
        dataset = test.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=test.column_names)
        train_dataset = dataset
        test_dataset = dataset
        return train_dataset, test_dataset
    else:
        def filter_func(x):
            return (0 < len(x["transcription"]) < max_ctx and
                    len(x["audio"]["array"]) > 0 and
                    len(x["audio"]["array"]) < max_ctx * 160)

        raw_train = load_dataset(
            "google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=streaming).take(1000)
        raw_test = load_dataset(
            "google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).take(100)

        raw_train = raw_train.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
        raw_test = raw_test.filter(filter_func).cast_column("audio", Audio(sampling_rate=sample_rate))
        train_dataset = raw_train.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_train.column_names)
        test_dataset = raw_test.map(lambda x: extract_features(x, tokenizer, **extract_args), remove_columns=raw_test.column_names)
        train_dataset.save_to_disk(cache_file_train) if save_dataset is True else None
        test_dataset.save_to_disk(cache_file_test) if save_dataset is True else None
        return train_dataset, test_dataset

#### -gates- 

class cgate(nn.Module):
    def __init__(n, dims, enabled=True):
        super().__init__()
        n.enabled = enabled
        if enabled:
            n.s_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.w_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.p_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.e_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.ph_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
            n.integ = nn.Linear(dims*5, dims)
            n.reset_parameters()
        
    def forward(n, x, enc):
        if not n.enabled:
            return None
        s_feat = enc.get("spectrogram", x)
        w_feat = enc.get("waveform", x)
        p_feat = enc.get("pitch", x)
        e_feat = enc.get("envelope", x)
        ph_feat = enc.get("phase", x)
        s = n.s_gate(x) * s_feat
        w = n.w_gate(x) * w_feat
        p = n.p_gate(x) * p_feat
        e = n.e_gate(x) * e_feat
        ph = n.ph_gate(x) * ph_feat
        comb = torch.cat([s, w, p, e, ph], dim=-1)
        return n.integ(comb)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class mgate(nn.Module):
    def __init__(n, dims, mem=64):
        super().__init__()
        n.mk = nn.Parameter(torch.randn(mem, dims))
        n.mv = nn.Parameter(torch.randn(mem, 1))
        n.mg = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        n.reset_parameters()

    def forward(n, x, cos=False):
        if cos:
            key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mk, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        else:
            key = F.softmax(torch.matmul(x, n.mk.transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        return 0.5 * (torch.sigmoid(n.mg(x)) + torch.sigmoid(torch.matmul(key, n.mv)))

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tmgate(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        n.dims=dims
        n.mkeys = {}

        n.xa_proj = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        n.pattern = lambda length: sinusoids(length, dims=dims)  
        n.primer = torch.ones(1, 512, dims)
        n.reset_parameters()

    def forward(n, x) -> torch.Tensor: 
        if x is None:
            cur = n.primer
            n.key = cur
        else:
            cur = n.pattern(x.shape[1]).to(device, dtype)
    
        n.mkeys["last"] = n.key
        cur = n.xa_proj(cur.mean(dim=1)) 

        for b in range(cur.size(0)):
            cur_xa = cur[b]
            score = -1.0
            best = None
            for last in n.mkeys.items():
                last = n.mkeys["last"]
                similarity = F.cosine_similarity(cur_xa, last, dim=0).mean()

                if similarity > score:
                    score = similarity
                    best = best

            gating_value = n.activation(torch.tensor(score))
            if gating_value > n.threshold and best is not None:
                n.key = cur
            else:
                n.key = last
            threshold = apply_ste_threshold(x, n.threshold)
        return threshold, n.key

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x, threshold)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste_threshold = StraightThroughThreshold.apply

def sinusoidz(length, dims):
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))
    pe = torch.zeros(length, dims)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class LinearGate(nn.Linear):
    def __init__(n, in_features, out_features, act="swish", norm_type=None, context = 4, num_types=4, top=4):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        n.context = context
        n.top = top
        n.num_types = num_types
        n.act=act

        n.gate = nn.Sequential(get_norm(norm_type, in_features), nn.Linear(in_features, out_features, bias=False), nn.Softmax(dim=-1))
        n.context = nn.Parameter(torch.ones(context), requires_grad=True)
        n.bias2 = nn.Parameter(torch.zeros(context, in_features), requires_grad=True)
        n.reset_parameters()

    def forward(n, x, top=4):
        x = x.unsqueeze(-2)
        x = x * (1 + get_activation(n.act)((rearrange(n.context, 'n -> 1 1 n 1') * n.gate(x).squeeze(-1)).mean(-1, keepdim=True)))
        x = x + n.bias2.unsqueeze(0)
        _, indices = torch.topk(x, n.top, dim=-2, sorted=False)
        x = torch.gather(x, -2, indices).mean(dim=-2)
        x = super().forward(x)
        return x

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class FeatureGate(nn.Module):
    def __init__(n, dims: int, expand: int, adapt_dim: int):
        super().__init__()

        n.steps = nn.ModuleList
        (
            [
            LinearGate(dims, dims, adapt_dim) 
            for _ in range(expand)
            ]
        )
        n.gating = nn.Linear(adapt_dim, expand)
        n.reset_parameters()

    def forward(n, xa, xb): # for audio features 
        scores = F.softmax(n.gating(xa), dim=-1)
        output = sum(scores[:, q].unsqueeze(1) * gate(xa, xb) for q, gate in enumerate(n.steps))
        return output

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class dgate(nn.Module):
    def __init__(n, dims: int, head: int, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        
        n.register_buffer('last_key', torch.zeros(1, dims))
        n.xa_proj = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        n.register_buffer('primer', torch.ones(1, 512, dims))
        n.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=True)
        n.reset_parameters()

    def forward(n, x: torch.Tensor): 
        device, dtype = x.device, x.dtype if x is not None else n.primer.device, n.primer.dtype
        if x is None:
            cur_representation = n.primer.to(device, dtype).mean(dim=1)
        else:
            pattern_tensor = n.pattern(x.shape[1]).to(device, dtype)
            cur_representation = pattern_tensor.mean(dim=1)

        cur_proj = n.xa_proj(cur_representation)
        expanded_last_key = n.last_key.expand(cur_proj.shape[0], -1) 
        similarity_scores = F.cosine_similarity(cur_proj, expanded_last_key, dim=-1).unsqueeze(-1)
        gating_value = n.activation(similarity_scores)
        decision = apply_ste_threshold(gating_value, n.threshold)
        n.last_key.copy_(cur_representation.detach()) 
        return decision, gating_value

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class m_gate(nn.Module):
    def __init__(n, dims, mem_size=64):
        super().__init__()

        n.m_key = nn.Parameter(torch.randn(mem_size, dims))
        n.m_val = nn.Parameter(torch.randn(mem_size, 1))
        n.gate_proj = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        n.reset_parameters()
            
    def forward(n, x):
        d_gate = torch.sigmoid(n.gate_proj(x))
        attention = torch.matmul(x, n.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, n.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class lgate(nn.Module):
    def __init__(n, dims):
        super().__init__()
        n.gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.reset_parameters()

    def forward(n, x):
        return n.gate(x)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tgate(nn.Module):
    def __init__(n, dims, num_types=2):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.reset_parameters()

    def forward(n, x):
        types = n.classifier(x)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        return  torch.sum(gates * types.unsqueeze(2), dim=-1)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class tgate_hybrid(nn.Module):
    def __init__(n, dims, num_types=10, k=2):
        super().__init__()
        n.num_types = num_types
        n.k = k
        
        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.sparse_classifier = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1))
        n.reset_parameters()

    def forward(n, x):
        soft_types = n.classifier(x)
        logits = n.sparse_classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        sparse_gates_values = F.softmax(top_k_logits, dim=-1)
        sparse_types = torch.zeros_like(soft_types)
        sparse_types.scatter_(-1, top_k_indices, sparse_gates_values)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        mixed_types = torch.sigmoid(n.alpha) * sparse_types + (1 - torch.sigmoid(n.alpha)) * soft_types
        return torch.sum(gates * mixed_types.unsqueeze(2), dim=-1)
    
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class tgate_conditional(nn.Module):
    def __init__(n, dims, num_types=2, k=1, use_sparse=False):
        super().__init__()
        n.num_types = num_types
        n.k = k
        n.use_sparse = use_sparse
        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        
        if n.use_sparse:
            n.classifier = nn.Linear(dims, num_types)
        else:
            n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.reset_parameters()

    def forward(n, x):
        if n.use_sparse:
            logits = n.classifier(x)
            top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
            sparse_gates_values = F.softmax(top_k_logits, dim=-1)
            types = torch.zeros_like(logits)
            types.scatter_(-1, top_k_indices, sparse_gates_values)
        else:
            types = n.classifier(x)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        return torch.sum(gates * types.unsqueeze(2), dim=-1)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class tgate_topk(nn.Module):
    def __init__(n, dims, num_experts=4, k=2):
        super().__init__()
        n.num_experts = num_experts
        n.k = k
        n.classifier = nn.Linear(dims, num_experts)
        n.experts = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_experts)])
        n.reset_parameters()

    def forward(n, x):
        logits = n.classifier(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        mask = torch.zeros_like(logits, requires_grad=False)
        mask.scatter_(-1, top_k_indices, 1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, top_k_indices, top_k_gates)
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=-1)
        return torch.sum(expert_outputs * gates.unsqueeze(2), dim=-1)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

class BiasingGateRefactored(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super(BiasingGateRefactored, n).__init__()
        n.memory_size = memory_size
        n.threshold = threshold
        
        n.register_buffer("mkeys_pattern", sinusoids(memory_size, dims))
        n.mkeys_bias = nn.Parameter(torch.randn(memory_size, head))
        
        n.xa_projection = nn.Linear(dims, dims)
        n.activation = nn.Sigmoid()
        
    def forward(n, x, xa) -> torch.Tensor:
        if x is None:
            return None
        
        xa_projected = n.xa_projection(xa.mean(dim=1))
        similarity = F.cosine_similarity(xa_projected.unsqueeze(1), n.mkeys_pattern.unsqueeze(0), dim=-1)
        
        best_scores, best_indices = torch.max(similarity, dim=1)
        best_bias_weights = n.mkeys_bias[best_indices]
        gating_value = n.activation(best_scores)
        
        mask = (gating_value > n.threshold).float().unsqueeze(-1)
        
        shot_bias = x
        scaled_bias = shot_bias * best_bias_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        final_bias = mask * scaled_bias + (1 - mask) * shot_bias
        return final_bias

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class SimpleMoE(nn.Module):
    def __init__(n, num_experts: int, expert_size: int, output_size: int):
        super().__init__()

        n.experts = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        n.gate = nn.Linear(expert_size, num_experts)

    def forward(n, x: Tensor) -> Tensor:
  
        gate_outputs = F.softmax(n.gate(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=1)
        return torch.einsum("ab,abc->ac", gate_outputs, expert_outputs)

class LessSimpleMoE(nn.Module):
    def __init__(n, dims, num_experts=5, num_modalities=5, k=1, memory_size=64, threshold=0.8):
        super().__init__()
        n.dims = dims
        n.num_experts = num_experts
        n.num_modalities = num_modalities
        n.k = k
        n.memory_size = memory_size
        n.threshold = threshold
        
        n.soft_router = nn.Sequential(nn.Linear(dims, num_experts), nn.Softmax(dim=-1))
        n.sparse_router = nn.Linear(dims, num_experts)

        n.m_key = nn.Parameter(torch.randn(memory_size, dims))
        n.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
        n.direct_gate = nn.Sequential(nn.Linear(dims, num_experts), nn.Sigmoid())

        n.experts = nn.ModuleList( [nn.Sequential(nn.Linear(dims, dims), nn.SiLU(), nn.Linear(dims, dims)) for _ in range(num_experts)] )
        
        n.fusion_gates = nn.ModuleList( [nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_modalities)] )
        n.fusion_projection = nn.Linear(dims * num_modalities, dims)
        
        n.register_buffer("alpha_moe", torch.tensor(0.5))
        n.register_buffer("alpha_fusion", torch.tensor(0.5))
        n.reset_parameters()
    
    def forward(n, x, enc):

        soft_expert_weights = n.soft_router(x)
        
        logits = n.sparse_router(x)
        top_k_logits, top_k_indices = torch.topk(logits, n.k, dim=-1)
        sparse_gates = F.softmax(top_k_logits, dim=-1)
        
        sparse_expert_weights = torch.zeros_like(logits)
        sparse_expert_weights.scatter_(-1, top_k_indices, sparse_gates)
        
        attention = torch.matmul(x, n.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(n.dims), dim=-1)

        memory_weights = torch.matmul(attention, n.m_val)
        memory_weights = F.softmax(memory_weights, dim=-1)

        mixed_expert_weights = (n.alpha_moe * sparse_expert_weights + (1 - n.alpha_moe) * soft_expert_weights)
        final_expert_weights = (mixed_expert_weights + memory_weights) / 2
        expert_outputs = torch.stack([expert(x) for expert in n.experts], dim=-1)
        moe_output = torch.sum(expert_outputs * final_expert_weights.unsqueeze(1), dim=-1)

        features = [enc.get("spectrogram", x), enc.get("waveform", x), enc.get("pitch", x), enc.get("envelope", x), enc.get("phase", x)]
        gates = [gate(x) for gate in n.fusion_gates]
        
        scaled_features = [gates[q] * features[q] for q in range(len(features))]
        fused_output = n.fusion_projection(torch.cat(scaled_features, dim=-1))
        
        return (n.alpha_fusion * fused_output) + ((1 - n.alpha_fusion) * moe_output)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class BiasingGateB(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int = 64, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        n.head = head
        n.memory_size = memory_size
        n.threshold = threshold
        n.mkeys = {}
        n.p = nn.Linear(dims, dims)
        n.tgate = nn.Sigmoid()

        n.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
        n.one_shot = OneShot(dims, head)
        n.reset_parameters()

        for _ in range(memory_size): # example
            pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
            bias_weight = OneShot(dims, head)
            n.mkeys[tuple(pattern.tolist())] = bias_weight

    def forward(n, x, xa) -> torch.Tensor:
        B, T, _ = x.shape
        input = n.p(x.mean(dim=1))
        batch_gate_biases = []
        for b in range(B):
            cur_input = input[b]
            score = -1.0
            best_match = None
            for pattern, gate_bias in n.mkeys.items():
                pattern_tensor = torch.tensor(pattern).to(cur_input.device)
                similarity = F.cosine_similarity(cur_input, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    best_match = gate_bias
            gating_value = n.tgate(score.unsqueeze(0))
            if gating_value > n.threshold and best_match is not None:
                batch_gate_biases.append(best_match.unsqueeze(0))
            else:
                batch_gate_biases.append(torch.zeros(1, n.head).to(x.device))
        return torch.cat(batch_gate_biases, dim=0)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class m2gate(nn.Module):
    def __init__(n, dims, mem=64, thresh=0.5):
        super().__init__()

        n.mkeys = nn.ParameterList([
            nn.Parameter(torch.randn(dims)),
            nn.Parameter(torch.randn(dims)),
        ])
        
        n.key_matrix = nn.Parameter(torch.randn(mem, dims))
        n.val_matrix = nn.Parameter(torch.randn(mem, 1))
        n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False).to(device)
        n.concat_layer = nn.Linear(2, 1, device=device, dtype=dtype)
        n.tgate_activation = nn.Sigmoid() 
        n.xa_projection = nn.Linear(dims, dims)

        n.register_buffer('previous_best_pattern', None)

    def forward(n, x):
        x_processed = n.xa_projection(x.mean(dim=1))
        skip_indicators = torch.ones(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None 

            for pattern_tensor in n.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)

                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                skip_indicators[b] = 1.0
            else:
                skip_indicators[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern 
        change_scores = torch.zeros(x_processed.size(0), dtype=torch.float32, device=x.device)

        previous_input_pattern_in_batch = None 
        
        for b in range(x_processed.size(0)):
            cur_x_element = x_processed[b]
            score = -1.0
            current_best_pattern = None

            for pattern_tensor in n.mkeys:
                similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)
                if similarity > score:
                    score = similarity
                    current_best_pattern = pattern_tensor

            if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
                change_scores[b] = 1.0
            else:
                change_scores[b] = 0.0
            previous_input_pattern_in_batch = current_best_pattern

        scalar = apply_ste_threshold(change_scores.unsqueeze(-1), n.threshold)
        key = F.softmax(torch.matmul(F.normalize(x_processed, p=2, dim=-1), F.normalize(n.key_matrix, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x_processed.shape[-1]), dim=-1)
        gate = n.concat_layer(torch.cat((torch.matmul(key, n.val_matrix),  n.mlp(x_processed)), dim=-1))
        return scalar, gate

class Memory(nn.Module):
    def __init__( n, new_dims: int, old_dims: int):
        super().__init__()

        n.new_dims = new_dims
        n.old_dims = old_dims
        n.new = nn.Linear(new_dims, old_dims)
        n.old = nn.Linear(old_dims, old_dims)
        n.activation = nn.Tanh()

    def forward(n, x: Tensor, y: Tensor) -> Tensor:
        mix = n.activation(n.new(x) + n.old(y))
        return mix

    def initialize_old(n, batch: int, device: torch.device) -> Tensor:
        old_dims = torch.zeros(batch, n.old_dims).to(device)
        return old_dims

class MixtureOfMemories(nn.Module):
    def __init__(n, dims, head, num_experts: int, expert_size: int, output_size: int):
        super().__init__()
        n.mems = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
        n.gate = nn.Linear(expert_size, num_experts)
        n.attention = nn.MultiheadAttention(dims, head)
        n.memory = Memory(dims, dims)
        n.layernorm = nn.LayerNorm(dims)

    def forward(n, x: Tensor, y: Tensor) -> Tensor:
        output, _ = n.attention(x, x, x)
        y = n.memory(output.mean(dim=0), y)
        gate_outputs = F.softmax(n.gate(x), dim=1)
        _outputs = torch.stack([mem(x) for mem in n.mems], dim=1)
        _output = torch.einsum("ab,abc->ac", gate_outputs, _outputs)
        output = n.layernorm(output + _output.unsqueeze(0))
        return output

class Transformer(nn.Module):
    def __init__( n, dims: int, head: int, num_experts: int, expert_size: int, num_layers: int):
        super().__init__()

        n.layers = nn.ModuleList([MixtureOfMemories( dims, head, num_experts, expert_size ) for _ in range(num_layers)])
        n.y = torch.zeros(1, dims)

    def forward(n, x: Tensor) -> Tensor:
        for layer in n.layers:
            return layer(x, n.y)
      
# class ContextualBiasingGate2(nn.Module):
#     def __init__(n, dims: int, head: int, memory_size: int, threshold: float = 0.8):
#         super().__init__()
#         n.dims = dims
#         n.head = head
#         n.memory_size = memory_size
#         n.threshold = threshold
#         n.mkeys = {}  # {pattern_embedding_tuple: bias_scalar_weight_tensor}
#         n.xa_projection = nn.Linear(dims, dims)
#         n.tgate_activation = nn.Sigmoid()
#         n.head_bias_weights = nn.Parameter(torch.ones(head))
#         n.pattern = lambda length, dims, max_tscale: sinusoids(length, dims)  
#         n.embedding = nn.Embedding(n.pattern, dims)
#         n.one_shot = OneShot(dims, head)
#         n.reset_parameters()

#         for _ in range(memory_size): # example
#             pattern = lambda length, dims, max_tscale: sinusoids(length, dims) 
#             bias_weight = OneShot(dims, head)
#             n.mkeys[tuple(pattern.tolist())] = bias_weight

#     def forward(n, shot_bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         x = n.xa_projection(x.mean(dim=1)) # (B, D)
#         last_x = []
#         previous_best = None 

#         for b in range(x.size(0)):
#             cur_x = x[b]
#             score = -1.0
#             current_best = None  

#             for pattern, bias in n.mkeys.items():
#                 pattern_tensor = torch.tensor(pattern).to(cur_x.device)
#                 similarity = F.cosine_similarity(cur_x, pattern_tensor, dim=0)

#                 if similarity > score:
#                     score = similarity
#                     current_best = bias

#             gating_value = n.tgate_activation(score.unsqueeze(0))
#             current_shot_bias = shot_bias[b] # (head, Q, K)

#             if gating_value > n.threshold and current_best is not None:
#                 if previous_best is None or not torch.equal(current_best, previous_best): #
#                     scaled_bias = current_shot_bias * (current_best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # (H,Q,K) * (H,1,1,1)
#                     last_x.append(scaled_bias)
#                 else: 
#                     last_x.append(current_shot_bias)
#             else: 
#                 last_x.append(current_shot_bias)
#             previous_best = current_best 
#         return torch.stack(last_x, dim=0) # (B, head, Q, K)

#     def reset_parameters(n) -> None:
#         for m in n.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0.0, std=0.01)        

# # Find the single best-matching memory key
# similarity = F.cosine_similarity(x.unsqueeze(1), n.m_key.unsqueeze(0), dim=-1)
# best_index = torch.argmax(similarity, dim=1) # (B,)

# # Retrieve the corresponding value from memory
# memory_weights = n.m_val[best_index] # (B, num_experts)

# # A simplified controller could use an RNN
# controller_output, _ = n.controller(x)

# # Use the controller output to generate read/write head
# read_head = n.read_head_proj(controller_output) # (B, memory_size)
# # The read head directly determines weights for memory access
# memory_weights = F.softmax(read_head, dim=-1)

# # A high-level router selects the relevant memory layer
# layer_scores = n.layer_router(x)

# # A layer-specific attention mechanism then retrieves the memory
# memory_weights_per_layer = [layer_attention(x) for layer_attention in n.layer_attentions]

# # Combine the outputs
# final_memory_weights = torch.sum(layer_scores.unsqueeze(-1) * torch.stack(memory_weights_per_layer, dim=1), dim=1)

# # Replace dot-product with a kernel function
# kernel_output = n.kernel_function(x, n.m_key)

# # Normalize to get weights
# memory_weights = kernel_output / torch.sum(kernel_output, dim=-1, keepdim=True)

# class SuperGateHybridMem(nn.Module):
#     def __init__(n, dims, num_experts=5, memory_size=64, **kwargs):
#         super().__init__()
#         # ... other initializations ...
#         n.memory_size = memory_size
#         n.m_key = nn.Parameter(torch.randn(memory_size, dims))
#         n.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
        
#         # Trainable mixing parameter for memory
#         n.alpha_mem = nn.Parameter(torch.tensor(0.5))

#     def forward(n, x, enc):
#         # ... other forward logic ...
        
#         # 1. Standard Attention-based Memory Gating
#         attention = torch.matmul(x, n.m_key.transpose(0, 1))
#         attention = F.softmax(attention / math.sqrt(n.dims), dim=-1)
#         attention_memory_weights = torch.matmul(attention, n.m_val)
        
#         # 2. Content-based Retrieval (Hard Selection)
#         similarity = F.cosine_similarity(x.unsqueeze(1), n.m_key.unsqueeze(0), dim=-1)
#         best_indices = torch.argmax(similarity, dim=1)
#         retrieval_memory_weights = n.m_val[best_indices]
        
#         # 3. Combine memory outputs
#         mixed_memory_weights = (torch.sigmoid(n.alpha_mem) * attention_memory_weights) + \
#                                ((1 - torch.sigmoid(n.alpha_mem)) * retrieval_memory_weights)
        
#         # Use mixed_memory_weights in the final combination
#         final_expert_weights = (mixed_expert_weights + mixed_memory_weights) / 2
        
#         # ... remaining forward logic ...
        
#         return final_output

# class SuperGateBiasedMem(nn.Module):
#     def __init__(n, dims, num_experts=5, memory_size=64, **kwargs):
#         super().__init__()
#         # ... other initializations ...
#         n.memory_size = memory_size
#         n.m_key = nn.Parameter(torch.randn(memory_size, dims))
#         n.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
        
#         # Projection for retrieved memory to combine with the query
#         n.mem_proj = nn.Linear(num_experts, dims)

#     def forward(n, x, enc):
#         # ... other forward logic ...
        
#         # Content-based Retrieval (Hard Selection)
#         similarity = F.cosine_similarity(x.unsqueeze(1), n.m_key.unsqueeze(0), dim=-1)
#         best_indices = torch.argmax(similarity, dim=1)
#         retrieved_val = n.m_val[best_indices]
        
#         # Use the retrieved memory as a bias for the attention query
#         biased_query = x + n.mem_proj(retrieved_val)
        
#         # Standard Attention-based Memory Gating, but with the biased query
#         attention = torch.matmul(biased_query, n.m_key.transpose(0, 1))
#         attention = F.softmax(attention / math.sqrt(n.dims), dim=-1)
#         memory_weights = torch.matmul(attention, n.m_val)
        
#         # ... remaining forward logic ...
        
#         return final_output

# class SuperGateConditionalMem(nn.Module):
#     def __init__(n, dims, num_experts=5, memory_size=64, threshold=0.8, **kwargs):
#         super().__init__()
#         # ... other initializations ...
#         n.memory_size = memory_size
#         n.m_key = nn.Parameter(torch.randn(memory_size, dims))
#         n.m_val = nn.Parameter(torch.randn(memory_size, num_experts))
#         n.threshold = threshold

#     def forward(n, x, enc):
#         # ... other forward logic ...
        
#         # Content-based Retrieval (Hard Selection)
#         similarity = F.cosine_similarity(x.unsqueeze(1), n.m_key.unsqueeze(0), dim=-1)
#         best_score, best_indices = torch.max(similarity, dim=1)
        
#         # Conditional logic to choose memory access method
#         if torch.mean(best_score) > n.threshold:
#             # Use content-based retrieval if a strong match is found
#             memory_weights = n.m_val[best_indices]
#         else:
#             # Fall back to standard attention
#             attention = torch.matmul(x, n.m_key.transpose(0, 1))
#             attention = F.softmax(attention / math.sqrt(n.dims), dim=-1)
#             memory_weights = torch.matmul(attention, n.m_val)
            
#         # ... remaining forward logic ...
        
#         return final_output

# class LiquidCell(nn.Module):

#     def __init__(n, new_dims: int, old_dims: int, dropout: float = 0.1, layer_norm: bool = True):
#         super(LiquidCell, n).__init__()
#         n.new_dims = new_dims
#         n.old_dims = old_dims
#         n.dropout = nn.Dropout(dropout)

#         n.w_in = nn.Linear(new_dims, old_dims)
#         n.w_h = nn.Linear(old_dims, old_dims)

#         n.layer_norm = (nn.LayerNorm(old_dims) if layer_norm else None)
#         n.activation = nn.Tanh()

#     def forward(n, x: Tensor, h: Tensor) -> Tensor:

#         new_h = n.activation(n.w_in(x) + n.w_h(h))

#         if n.layer_norm:
#             new_h = n.layer_norm(new_h)

#         new_h = n.dropout(new_h)
#         return new_h

#     def initialize_x(n, batch_size: int, device: torch.device) -> Tensor:
#         x = torch.zeros(batch_size, n.old_dims).to(device)
#         return x

### mini attentions

class CausalAttention(nn.Module):
    def __init__(n, dims, head, ctx):
        super().__init__()
        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims * 2, bias=False)
        n.n = nn.LayerNorm(dims) 
        n.register_buffer('mask', torch.triu(torch.ones(ctx, ctx), diagonal=1)) 
        n.reset_parameters()
    def forward(n, x):
        k, v = n.k(n.n(x)).chunk(2, dim=-1)
        q = n.q(n.n(x))
        qk = q @ k.transpose(1, 2) 
        qk.masked_fill_(n.mask.bool()[:x.shape[1], :x.shape[1]], -torch.inf) 
        w = torch.softmax(qk / k.shape[-1]**0.5, dim=-1)
        return  w @ v

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(n, dims, head, ctx):
        super().__init__()
        n.head = nn.ModuleList([CausalAttention(dims, head, ctx) for _ in range(head)])
    def forward(n, x):
        return torch.cat([head(x) for head in n.head], dim=-1)   

class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        n.head  = head
        n.hdim  = dims // head
        n.scale = scale                      
        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)
    def forward(n, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor | None:
        B, Q, _ = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, Q, n.head, n.hdim).transpose(1,2)  
        k = n.k(xa).view(B, K, n.head, n.hdim).transpose(1,2)
        return (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.hdim) 

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class ContextualBias(nn.Module):
    def __init__(n, dims: int, head: int, memory_size: int, threshold: float = 0.8):
        super().__init__()
        n.dims = dims
        n.head = head
        n.hdim = dims // head

        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)
        n.v = nn.Linear(dims, dims)
        n.one_shot = OneShot(dims, head)
        n.biasing_gate = BiasingGate(dims, head)
        n.reset_parameters()

    def forward(n, x, xa) -> torch.Tensor:
        B, Q, _ = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, Q, n.head, n.hdim).transpose(1,2)
        k = n.k(x if xa is None else xa).view(B, K, n.head, n.hdim).transpose(1,2)
        v = n.v(x if xa is None else xa).view(B, K, n.head, n.hdim).transpose(1,2)
        qk = (q @ k.transpose(-1, -2)) / math.sqrt(n.hdim) # (B, H, Q, K)
        x = n.one_shot(x, xa)
        if x is not None:
            bias = n.biasing_gate(x, xa)
            qk = qk + bias
        w = F.softmax(qk, dim=-1)
        return (w @ v).transpose(1, 2).reshape(B, Q, n.dims)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class CuriosityHead(nn.Module):
    def __init__(n, d, h, bias=True, memory_size=10, cb_threshold=0.7):
        super().__init__()
        n.h  = h              # base head
        n.dh = d // h
        n.qkv = nn.Linear(d, d * 3, bias=bias)
        n.qkv_aux = nn.Linear(d, d * 3, bias=bias)  # curiosity head
        n.o  = nn.Linear(d, d, bias=bias)
        n.g  = nn.Parameter(torch.zeros(h))         # per-head gate logit
        n.contextual_biasing_gate = BiasingGateB(d, h, memory_size, cb_threshold)
        n.reset_parameters()

    def split(n, x):
        b, t, _ = x.shape
        return x.view(b, t, n.h, n.dh).transpose(1, 2)  # b h t dh

    def merge(n, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(n, x, xa, mask=None):
        q, k, v   = n.qkv(x).chunk(3, -1)
        qa, ka, va = n.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(n.split, (q, k, v))
        qa, ka, va = map(n.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / n.dh**0.5      # b h t t
        dots_aux  = (q @ ka.transpose(-2, -1)) / n.dh**0.5     # b h t ta
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)

        h_main = p  @ v                       # b h t dh
        h_aux  = pa @ va                      # b h t dh
        contextual_gate_bias = n.contextual_biasing_gate(x) # (B, H)
        g_biased_logits = n.g + contextual_gate_bias # (B, H)
        g = torch.sigmoid(g_biased_logits).view(x.size(0), -1, 1, 1) # b h 1 1 broadcast
        out = n.merge(h_main * (1 - g) + h_aux * g)
        return n.o(out)

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

# ... (ContextualBiasingGate adapted to return an index or a preference vector) ...

# class LuckyGenesWithContextualSelection(nn.Module):
#     def __init__(n, d, h, population_size=8, mutation_rate=0.15, crossover_rate=0.3,
#                  cb_memory_size=10, cb_threshold=0.7):
#         super().__init__()
#         # ... (LuckyGenes init as before) ...
#         n.contextual_selection_gate = ContextualBiasingGateForSelection(d, population_size, cb_memory_size, cb_threshold)

#     def forward(n, x, xa, mask=None):
#         # ... (attention qkv, split, dots, etc. as before) ...

#         # Contextual selection of gate strategy
#         # The CBG returns a tensor (B, pop_size) representing preferences/logits for each population member
#         contextual_preferences = n.contextual_selection_gate(x) # using x as context
        
#         # Decide which gate to use
#         # This could be a softmax over preferences to get a weighted average of 'g's,
#         # or a direct selection based on highest preference.
        
#         # Simple selection: Choose the population member indicated by the CBG's output
#         # (Assuming CBG returns an index or a one-hot vector for the best match)
#         # For simplicity, let's assume CBG output (B, H) and we want one 'g' (H,)
#         # This part needs careful design for mapping B-level context to a single 'g' or
#         # dynamically combining population members.
        
#         # A more robust approach might be to have CBG return a *modification* to the `fitness` vector
#         # or a "selection pressure" vector (B, pop_size)
        
#         # Let's consider a simplified version where CBG outputs a 'suggested_idx' per batch item
#         # This would require adapting CBG's output to be an index or a probability distribution.

#         # Let's adapt CBG to output a (B, population_size) tensor of logits for selection
#         cb_selection_logits = n.contextual_biasing_gate_output_selector(x) # (B, pop_size)
        
#         # Combine CBG selection logits with global fitness
#         # This allows the GA's fitness to interact with contextual preference
#         # We need a 'g' for each batch item, so we can't just pick one `best_idx`.
        
#         # Option A: Dynamically blend population members based on context
#         selection_weights = F.softmax(cb_selection_logits, dim=-1) # (B, pop_size)
        
#         # Get all population gates, unsqueeze for batch dimension: (1, pop_size, H)
#         all_gates = torch.stack(list(n.population), dim=0).unsqueeze(0) 
        
#         # Compute a weighted average of gates for each batch item: (B, H)
#         g_for_batch = (selection_weights.unsqueeze(-1) * all_gates).sum(dim=1) # (B, H)

#         # Apply genetic gate
#         g = torch.sigmoid(g_for_batch).view(x.size(0), -1, 1, 1) # b h 1 1 broadcast
#         out = n.merge(h_main * (1 - g) + h_aux * g)
        
#         # ... (fitness update and evolution logic as before) ...
#         # Fitness update would need to consider which members contributed to 'g_for_batch'
#         # This makes the fitness update more complex (e.g., fractional fitness contribution)

#         return n.o(out)

# ... (ContextualBiasingGate adapted to return a scalar fitness boost for the chosen gate) ...

# class LuckyGenesWithContextualFitness(nn.Module):
#     def __init__(n, d, h, population_size=8, mutation_rate=0.15, crossover_rate=0.3,
#                  cb_memory_size=10, cb_threshold=0.7):
#         super().__init__()
#         # ... (LuckyGenes init as before) ...
#         n.contextual_fitness_booster = ContextualBiasingGateForFitness(d, cb_memory_size, cb_threshold)

#     def forward(n, x, xa, mask=None):
#         # Pick best gate from cur population (initial GA choice)
#         best_idx = torch.argmax(n.fitness)
#         g = torch.sigmoid(n.population[best_idx])
        
#         # ... (Normal attention computation) ...

#         # Apply genetic gate
#         g = g.view(1, -1, 1, 1)
#         out = n.merge(h_main * (1 - g) + h_aux * g)
        
#         # Update fitness of cur best based on usage
#         if n.training:
#             n.fitness[best_idx] += 1.0  # Simple fitness: usage count

#             # Get contextual fitness boost/penalty
#             # Assuming CBG returns a (B,) tensor of boosts. We average or sum.
#             contextual_boosts = n.contextual_fitness_booster(x) # using x as context
#             # Let's say CBG returns a (B,) tensor, we'll sum them for the cur best_idx
#             n.fitness[best_idx] += contextual_boosts.sum().item() # Or mean, or another aggregation

#             # Evolve every N steps
#             if n.generation % 100 == 0:
#                 n.evolve_population()
            
#         return n.o(out)

# ... (ContextualBiasingGate adapted to return mutation/crossover rate modifications) ...

# class LuckyGenesWithContextualGA(nn.Module):
#     def __init__(n, d, h, population_size=8, mutation_rate=0.15, crossover_rate=0.3,
#                  cb_memory_size=10, cb_threshold=0.7):
#         super().__init__()
#         # ... (LuckyGenes init as before) ...
#         n.contextual_ga_modulator = ContextualBiasingGateForGARates(d, cb_memory_size, cb_threshold)

#     def evolve_population(n, cur_context_mod_rates): # Pass context-dependent rates
#         # ... (tournament selection) ...
#         # Use cur_context_mod_rates for n.mut_rate and n.cross_rate
#         # Example:
#         # child = n.crossover(parent1, parent2, cur_cross_rate) # Pass rate
#         # if torch.rand(1) < cur_mut_rate: # Use rate
#         # ...
#         pass # Requires more significant modification to crossover/evolve_population

#     def forward(n, x, xa, mask=None):
#         # ... (forward pass logic) ...
        
#         if n.training:
#             # ... (fitness update) ...

#             # Get contextual modulation for GA rates
#             mod_rates = n.contextual_ga_modulator(x) # (B, 2) for mut_rate, cross_rate
#             # Take average across batch or use a specific element's context
#             cur_mut_rate = (n.mut_rate + mod_rates[:, 0].mean()).clamp(0.01, 0.5) # Example clamping
#             cur_cross_rate = (n.cross_rate + mod_rates[:, 1].mean()).clamp(0.1, 0.9)

#             # Evolve every N steps
#             if n.generation % 100 == 0:
#                 n.evolve_population(cur_mut_rate, cur_cross_rate) # Pass rates
                
#         return n.o(out)

# # Example usage
# d_model = 128
# head = 8
# memory_size = 10
# cb_threshold = 0.75

# curiosity_head = CuriosityHead(d_model, head, memory_size=memory_size, cb_threshold=cb_threshold)

# # Create dummy input tensors
# batch_size = 2
# seq_len_x = 20
# seq_len_xa = 30
# x_input = torch.randn(batch_size, seq_len_x, d_model)
# xa_input = torch.randn(batch_size, seq_len_xa, d_model)

# # Forward pass
# output = curiosity_head(x_input, xa_input)

# print(f"Output shape of CuriosityHead with contextual bias: {output.shape}")

# # Example usage
# input_dim = 128
# head = 4
# q_len = 10 # Query sequence length
# k_len = 20 # Key/Value sequence length
# memory_size = 5
# threshold = 0.7

# attention_module = MainAttentionWithContextualBias(input_dim, head, q_len, k_len, memory_size, threshold)

# # Create dummy input tensors
# query_input = torch.randn(1, q_len, input_dim)
# xa_input = torch.randn(1, k_len, input_dim)

# # Forward pass
# output = attention_module(query_input, xa_input)

# print(f"Output shape of attention with contextual bias: {output.shape}")

def get_encoder(mels = None, input_dims = None, dims = None, head=None, act="relu", downsample = True, target_length = False, attend_pitch=False, feature = "spectrogram") -> nn.Module:

    if feature == "spectrogram":
        return FEncoder(mels, dims, head, act, feature = "spectrogram")
    elif feature == "waveform":
        return WEncoder(input_dims, dims, head, act, downsample, target_length, feature = "waveform")
    elif feature == "pitch":
        return PEncoder(input_dims, dims, head, act, attend_pitch, feature = "pitch")
    else:
        raise ValueError(f"Unknown feature type: {feature}")

@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(n, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        pad_token_id = getattr(n.tokenizer, 'pad_token_id', 0)
        bos_token_id = getattr(n.tokenizer, 'bos_token_id', 1)
        eos_token_id = getattr(n.tokenizer, 'eos_token_id', 2)

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

            elif key in ["spectrogram", "waveform", "pitch", "harmonic", "aperiodic", "pitch_tokens", "f0", "phase", "crepe_time", "crepe_frequency", "crepe_confidence", "crepe_activation", "dummy"]:

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

                    # if pad_item == "spectrogram":
                    #     pad_item = FEncode(pad_item)

                    padded.append(pad_item)
                batch[key] = torch.stack(padded)

        return batch

def levenshtein(reference_words, hypothesis_words):
    m, n = len(reference_words), len(hypothesis_words)
    dist_matrix = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for q in range(m+1):
        dist_matrix[q][0] = q
    for k in range(n+1):
        dist_matrix[0][k] = k
    for q in range(1, m+1):
        for k in range(1, n+1):
            if reference_words[q-1] == hypothesis_words[k-1]:
                dist_matrix[q][k] = dist_matrix[q-1][k-1]
            else:
                substitution = dist_matrix[q-1][k-1] + 1
                insertion = dist_matrix[q][k-1] + 1
                deletion = dist_matrix[q-1][k] + 1
                dist_matrix[q][k] = min(substitution, insertion, deletion)
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

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None, compute_result: bool = False):
    
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(q) for q in seq if q not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(q) for q in ids if q not in (-100, pad_token_id, bos_token_id, eos_token_id)]

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if not isinstance(pred_ids, torch.Tensor):
        pred_ids = torch.tensor(pred_ids)

    label_ids = clean(label_ids)
    pred_ids = clean(pred_ids)
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(label_ids)

    if print_pred:
        for q in range(min(num_samples, len(pred_ids))):

            print(f"Pred tokens: {pred_ids[q]}")
            print(f"Label tokens: {label_ids[q]}")
            print(f"Pred: '{pred_str[q]}'")
            print(f"Label: '{label_str[q]}'")
            print("-" * 40)

    wer = wer_batch(label_str, pred_str)
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
        # jump_logs = model.processor.res.dsl.logs
    else:
        trainable_params = 0.0
        efficiency_score = 0.0
        # jump_logs = None

    return {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
        # "jump_logs": jump_logs["jumps"][0],
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    h = torch.zeros(N // 2 + 1, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.irfft(xf * h, n=N)

def analytic_signal(x):
    return x + 1j * hilbert_transform(x)

def hilbert_transform_2d(x, dim=-1):
    N = x.shape[dim]
    if dim == -1 or dim == len(x.shape) - 1:
        xf = torch.fft.rfft(x)
    else:
        xf = torch.fft.rfft(x, dim=dim)
    h_shape = [1] * len(x.shape)
    h_shape[dim] = N // 2 + 1
    h = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
    if dim == -1 or dim == len(x.shape) - 1:
        if N % 2 == 0:
            h[..., 0] = h[..., -1] = 1
            h[..., 1:-1] = 2
        else:
            h[..., 0] = 1
            h[..., 1:] = 2
    else:
        pass
    return torch.fft.irfft(xf * h, n=N, dim=dim)

def hilbert_transform_true_2d(x):
    xf = torch.fft.rfft2(x)
    h1, h2 = torch.meshgrid(
        torch.fft.rfftfreq(x.shape[-2]) * 2 - 1,
        torch.fft.rfftfreq(x.shape[-1]) * 2 - 1,
        indexing='ij')
    h = -1j / (math.pi * (h1 + 1j*h2))
    h[0, 0] = 0 
    return torch.fft.irfft2(xf * h.to(x.device))

def process_spectrogram_with_hilbert(spec):
    analytic = spec + 1j * hilbert_transform(spec)
    envelope = torch.abs(analytic)
    phase = torch.angle(analytic)
    return envelope, phase

class MyModel(nn.Module):
    def __init__(n, layer, dims, head, act_fn):
        super().__init__()
        n.layer = layer 

        n.layers = nn.ModuleList()
        for q in range(layer):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if q % 2 == 0 else None,
                "res1": residual(dims, head, layer, act_fn),
                "res2": residual(dims, head, layer, act_fn),
                "res3": residual(dims, head, layer, act_fn),      
            }
            if q == 5:
                layer_dict['special_attention'] = attentiona(dims, head)
            n.layers.append(nn.ModuleDict(layer_dict))
        
        # n.attention_auxiliary = attentiona(dims, head) 

    def forward(n, x, xa, enc=None, sequential=False, modal=False, blend=False, kv_cache=None) -> torch.Tensor:    
   
        if n.layer > 0:
            cur_dict = n.layers[0]
            xa = cur_dict['res1'](xa)

        if n.layer > 1:
            cur_dict = n.layers[1]
            a = cur_dict['res2'](x)

        if n.layer > 2:
            cur_dict = n.layers[2]
            b = cur_dict['res3'](x, xa, None)

        if n.layer > 3:
            cur_dict = n.layers[3]
            xm = cur_dict['res1'](torch.cat([x, xa], dim=1))

        if n.layer > 4:
            cur_dict = n.layers[4]
            c = cur_dict['res2'](xm[:, :x.shape], xm[:, x.shape:], mask=None)

        if n.layer > 5:
            cur_dict = n.layers[5]

            x = cur_dict['res3'](a + b + c) 
            
            x = cur_dict['res3'](x, xa, None)

            x = cur_dict['lna'](x)

            if 'special_attention' in cur_dict:
                special_attn_output = cur_dict['special_attention'](x, xa=x, mask=None)
                x = x + special_attn_output

            aux_attn_output = n.attention_auxiliary(x, xa=x, mask=None)
            x = x + aux_attn_output

        for q in range(6, n.layer):
            cur_dict = n.layers[q]
            x = cur_dict['res1'](x)
            x = cur_dict['lna'](x)

        return x

        # n.layers = nn.ModuleList()
        # for _ in range(layer):
        #     n.layers.append(nn.ModuleList({
        #         'lna': nn.LayerNorm(dims),
        #         'lnb': nn.LayerNorm(dims),
        #         "res1": residual(dims, head, layer, act_fn),
        #         "res2": residual(dims, head, layer, act_fn),
        #         "res3": residual(dims, head, layer, act_fn),      
                    
        #         }))

        # if n.layer > 0:
        #     res = n.layers[0]
        #     xa = res['res1'](xa) 

        # if n.layer > 1:
        #     res = n.layers[1]
        #     a  = res['res2'](x, mask=mask)

        # if n.layer > 4:
        #     res = n.layers[4]
        #     b = res['res3'](x, xa, None)

        # # for q in range(3, n.layer):
        # #     res = n.layers[q]
        #     xm = res['res4'](torch.cat([x, xa], dim=1))
        #     c  = res['res4'](xm[:, :x.shape], xm[:, x.shape:], mask=None)

        # for q in range(4, n.layer):
        #     res = n.layers[q]
        #     x = res['res3'](a + b + c) 
        #     x = res['res3'](x, xa, None)
        #     x = res['lna'](x)

class attention_a(nn.Module):
    def __init__(n, dims: int, head: int, layer):
        super().__init__()
        n.head = head
        n.dims = dims
        n.head_dim = dims // head

        n.q = nn.Linear(dims, dims) 
        n.kv = nn.Linear(dims, dims * 2, bias=False)
        n.out = nn.Linear(dims, dims)
        n.lna = nn.LayerNorm(dims) 
        n.lnb = nn.LayerNorm(dims // head) 

    def forward(n, x, xa = None, mask = None):
        q = n.q(x)
        k, v = n.kv(n.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = n.head), (q, k, v))
        a = scaled_dot_product_attention(n.lnc(q), n.lnd(k), v, is_causal=mask is not None and q.shape[2] > 1)
        wv = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        out = n.out(wv)
        return out

class attentio0(nn.Module):
    def __init__(n, dims: int, head: int, layer):
        super().__init__()
        n.head = head
        n.dims = dims
        n.head_dim = dims // head

        n.pad_token = 0
        n.zmin = 1e-6
        n.zmax = 1e-5     
        n.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        n.taylor_expand_fn = partial(second_taylor)
        n.q = nn.Linear(dims, dims) 
        n.kv = nn.Linear(dims, dims * 2, bias=False)
        n.out = nn.Linear(dims, dims)

        n.lna = nn.LayerNorm(dims) 
        n.lnb = nn.LayerNorm(dims // head) 

        # n.rotary_emb = RotaryEmbedding(dims // head)

       # print(f"x, {x.shape}, xa, {xa.shape if xa is not None else None}, mask {mask.shape if mask is not None else None}")
        # zero = n.zero
    def forward(n, x, xa = None, mask = None,  positions = None):

        q = n.q(x)
        k, v = n.kv(n.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = n.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', n.lnb(q), n.lnb(k)) * scale 

        if mask is not None:
            mask=mask[:q.shape[2], :q.shape[2]]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(n.fzero), n.minz, n.maxz)
        zscale[token_ids.float() == n.pad_token] = fzero
        
        if xa is not None:
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        qk = taylor_softmax(qk, order=2)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = n.out(wv)
        return out

class attentio0b(nn.Module):
    def __init__(n, dims: int, head: int, layer):
        super().__init__()
        n.head = head
        n.dims = dims
        n.head_dim = dims // head

        n.pad_token = 0
        n.zmin = 1e-6
        n.zmax = 1e-5     
        n.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        n.taylor_expand_fn = partial(second_taylor)
        n.q = nn.Linear(dims, dims) 
        n.kv = nn.Linear(dims, dims * 2, bias=False)
        n.out = nn.Linear(dims, dims)

        n.lna = nn.LayerNorm(dims) 
        n.lnb = nn.LayerNorm(dims // head) 

    def forward(n, x, xa = None, mask = None,  positions = None):

        q = n.q(x)
        k, v = n.kv(n.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = n.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', n.lnb(q), n.lnb(k)) * scale 

        if mask is not None:
            mask=mask[:q.shape[2], :q.shape[2]]

        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(n.fzero), n.minz, n.maxz)
        zscale[token_ids.float() == n.pad_token] = fzero
        
        if xa is not None:
            qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

        qk = qk * zscale.unsqueeze(-2)
        qk = F.softmax(qk, dim=-1)
        # qk = taylor_softmax(qk, order=2)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = n.out(wv)
        return out

class attentionz(nn.Module):
    def __init__(n, dims: int, head: int, layer):
        super().__init__()
        n.head = head
        n.dims = dims
        n.head_dim = dims // head

        n.pad_token = 0
        n.zmin = 1e-6
        n.zmax = 1e-5     
        n.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)
        n.taylor_expand_fn = partial(second_taylor)
        n.q = nn.Linear(dims, dims) 
        n.kv = nn.Linear(dims, dims * 2, bias=False)
        n.out = nn.Linear(dims, dims)

        n.lna = nn.LayerNorm(dims) 
        n.lnb = nn.LayerNorm(dims // head) 

        # n.rotary_emb = RotaryEmbedding(dims // head)

       # print(f"x, {x.shape}, xa, {xa.shape if xa is not None else None}, mask {mask.shape if mask is not None else None}")
        # zero = n.zero

def forward_revised(n, x, xa = None, mask = None):
    q = n.q(x)
    k, v = n.kv(n.lna(x if xa is None else xa)).chunk(2, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = n.head), (q, k, v))

    q_expanded = second_taylor(q)
    k_expanded = second_taylor(k)
    dk_expanded = k_expanded.shape[-1]
    scale_factor = dk_expanded ** -0.5
    qk_logits = torch.einsum('b h k d, b h q d -> b h k q', n.lnb(q_expanded), n.lnb(k_expanded)) * scale_factor

    if mask is not None:
        seq_len_q, seq_len_k = qk_logits.shape[-2:]
        causal_mask = torch.ones(seq_len_q, seq_len_k, device = qk_logits.device, dtype = torch.bool).triu(seq_len_k - seq_len_q + 1)
        qk_logits = qk_logits.masked_fill(causal_mask, -torch.finfo(qk_logits.dtype).max)

    qk_weights = taylor_softmax(qk_logits, order=2)
    wv = torch.einsum('b h k q, b h q d -> b h k d', qk_weights, v)
    wv = rearrange(wv, 'b h c d -> b c (h d)')
    out = n.out(wv)
    return out

# class mgate(nn.Module):
#     def __init__(n, dims, mem=64, thresh=0.5):
#         n.key = nn.Parameter(mem, dims)
#         n.val = nn.Parameter(torch.randn(mem, 1))
#         n.mlp = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
#         n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False)
#         n.concat = nn.Linear(2,1, device=device, dtype=dtype)

#     def forward(n, x):
#         key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.key, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
#         x = n.concat(torch.cat((torch.matmul(key, n.val),  n.mlp(x)), dim=-1))
#         threshold = apply_ste_threshold(x, n.threshold)
#         return threshold, x

# class mgate(nn.Module):
#     def __init__(n, dims, mem=64, thresh=0.5):
#          # x: batch,ctx,dims
#         n.mkeys = {}
#         n.key = nn.Parameter(mem, dims, requires_grad=False)
#         n.val = nn.Parameter(torch.randn(mem, 1))
#         n.mlp = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
#         n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False)
#         n.concat = nn.Linear(2,1, device=device, dtype=dtype)

#     def forward(n, x): # x: batch,ctx,dims
#          # x: batch,ctx,dims
#         x = n.xa_projection(x.mean(dim=1))
#         last_x = []
#         previous_best = None 

#         for b in range(x.size(0)):
#             cur_x = x[b]
#             score = -1.0
#             current_best = None  

#             for pattern, bias in n.mkeys.items():
#                 pattern_tensor = torch.tensor(pattern).to(cur_x.device)
#                 similarity = F.cosine_similarity(cur_x, pattern_tensor, dim=0)

#                 if similarity > score:
#                     score = similarity
#                     current_best = bias

#             gating_value = n.tgate_activation(score.unsqueeze(0))
#             current_key = n.key

#             if gating_value > n.threshold and current_best is not None:
#                 if previous_best is None or not torch.equal(current_best, previous_best):
#                     key = current_key * (current_best)
#                     last_x.append(key)
#                 else: 
#                     last_x.append(current_key)
#             else: 
#                 last_x.append(current_key)
#             previous_best = current_best 
#             n.key = current_key 

#         key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.key, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
#         gate = n.concat(torch.cat((torch.matmul(key, n.val),  n.mlp(x)), dim=-1))
#         threshold = apply_ste_threshold(x, n.threshold)
#         return threshold, gate

        # return torch.stack(last_x, dim=0) # (B, head, Q, K)

# class mgate(nn.Module):
#     def __init__(n, dims, mem=64, thresh=0.5):
#         super().__init__()
#         n.mkeys = nn.ParameterDict()

#         n.key_matrix = nn.Parameter(torch.randn(mem, dims))
#         n.val_matrix = nn.Parameter(torch.randn(mem, 1))
#         n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
#         n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False)
#         n.concat_layer = nn.Linear(2, 1)

#         n.xa_projection = nn.Linear(dims, dims)
#         n.tgate_activation = nn.Sigmoid()

#     def forward(n, x):
#         x = n.xa_projection(x.mean(dim=1)) 

#         binary_decisions = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

#         previous_input_pattern = None 

#         for b in range(x.size(0)):
#             cur_x = x[b]
#             score = -1.0
#             current_input_pattern = None

#             for pattern_key, pattern_tensor in n.mkeys.items():
#                 similarity = F.cosine_similarity(cur_x, pattern_tensor.squeeze(), dim=0)

#                 if similarity > score:
#                     score = similarity
#                     current_input_pattern = pattern_tensor

#             if previous_input_pattern is None or not torch.equal(current_input_pattern, previous_input_pattern):
#                 binary_decisions[b] = True
#             else:
#                 binary_decisions[b] = False

#             previous_input_pattern = current_input_pattern 
        
#         key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.key_matrix, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
#         gate = n.concat_layer(torch.cat((torch.matmul(key, n.val_matrix),  n.mlp(x)), dim=-1))
#         threshold_output = apply_ste_threshold(x, n.threshold)

#         return binary_decisions, threshold_output, gate 

# def apply_ste_threshold(x, threshold):
#     return (x > threshold).float().detach() + (x - x.detach())

# class mgate(nn.Module):
#     def __init__(n, dims, mem=64, thresh=0.5):
#         super().__init__()

#         n.mkeys = nn.ParameterList([
#             nn.Parameter(torch.randn(dims)),
#             nn.Parameter(torch.randn(dims)),
#         ])
        
#         n.key_matrix = nn.Parameter(torch.randn(mem, dims))
#         n.val_matrix = nn.Parameter(torch.randn(mem, 1))
#         n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
#         n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False).to(device)
#         n.concat_layer = nn.Linear(2, 1, device=device, dtype=dtype)
#         n.tgate_activation = nn.Sigmoid() 
#         n.xa_projection = nn.Linear(dims, dims)

#         n.register_buffer('previous_best_pattern', None)

#     def forward(n, x):
#         x_processed = n.xa_projection(x.mean(dim=1))
#         skip_indicators = torch.ones(x_processed.size(0), dtype=torch.float32, device=x.device)

#         previous_input_pattern_in_batch = None 
#         for b in range(x_processed.size(0)):
#             cur_x_element = x_processed[b]
#             score = -1.0
#             current_best_pattern = None 

#             for pattern_tensor in n.mkeys:
#                 similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)

#                 if similarity > score:
#                     score = similarity
#                     current_best_pattern = pattern_tensor

#             if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
#                 skip_indicators[b] = 1.0
#             else:
#                 skip_indicators[b] = 0.0
#             previous_input_pattern_in_batch = current_best_pattern 
#         change_scores = torch.zeros(x_processed.size(0), dtype=torch.float32, device=x.device)

#         previous_input_pattern_in_batch = None 
        
#         for b in range(x_processed.size(0)):
#             cur_x_element = x_processed[b]
#             score = -1.0
#             current_best_pattern = None

#             for pattern_tensor in n.mkeys:
#                 similarity = F.cosine_similarity(cur_x_element, pattern_tensor, dim=0)
#                 if similarity > score:
#                     score = similarity
#                     current_best_pattern = pattern_tensor

#             if previous_input_pattern_in_batch is None or not torch.equal(current_best_pattern, previous_input_pattern_in_batch):
#                 change_scores[b] = 1.0
#             else:
#                 change_scores[b] = 0.0
#             previous_input_pattern_in_batch = current_best_pattern

#         scalar = apply_ste_threshold(change_scores.unsqueeze(-1), n.threshold)
#         key = F.softmax(torch.matmul(F.normalize(x_processed, p=2, dim=-1), F.normalize(n.key_matrix, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x_processed.shape[-1]), dim=-1)
#         gate = n.concat_layer(torch.cat((torch.matmul(key, n.val_matrix),  n.mlp(x_processed)), dim=-1))
#         return scalar, gate

# class rotary(nn.Module):
#     def __init__(n, dims, head):
#         super(rotary, n).__init__()
#         n.dims = dims
#         n.head = head
#         n.head_dim = dims // head
#         n.taylor_order = 10

#         n.theta = nn.Parameter((torch.tensor(360000, device=device, dtype=dtype)), requires_grad=False)  
#         n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

#     def _compute_freqs_base(n):
#         mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
#         return 200 * mel_scale / 1000 

#     def forward(n, x) -> torch.Tensor:
#         positions = (torch.arange(0, x.shape[2], device=x.device))
#         freqs = (n.theta / 220.0) * n.freqs_base
#         freqs = positions[:, None] * freqs 
#         freqs_rescaled = (freqs + torch.pi) % (2 * torch.pi) - torch.pi 

#         with torch.autocast(device_type="cuda", enabled=False):
#             cos = vectorized_taylor_cosine(freqs_rescaled, order=n.taylor_order)
#             sin = vectorized_taylor_sine(freqs_rescaled, order=n.taylor_order)
#             rotary_dim = cos.shape[-1] 
#             x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
#             x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
#             x_embed = torch.cat([x_embed, x_pass], dim=-1)
#             return x_embed.type_as(x)

# class rotary(nn.Module):
#     def __init__(n, dims, head):
#         super(rotary, n).__init__()
#         n.dims = dims
#         n.head = head
#         n.head_dim = dims // head
#         n.taylor_order = 5

#         n.theta = nn.Parameter((torch.tensor(1600, device=device, dtype=dtype)), requires_grad=False)  
#         n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

#     def _compute_freqs_base(n):
#         mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
#         return 200 * mel_scale / 1000 

#     def forward(n, x) -> torch.Tensor:

#         positions = (torch.arange(0, x.shape[2], device=x.device))
#         freqs = (n.theta / 220.0) * n.freqs_base
#         freqs = positions[:, None] * freqs 

#         freqs = (freqs + torch.pi) % (2 * torch.pi) - torch.pi

#         with torch.autocast(device_type="cuda", enabled=False):
#             cos = taylor_cosine(freqs, order=n.taylor_order)
#             sin = taylor_sine(freqs, order=n.taylor_order)
#             rotary_dim = cos.shape[-1] 
#             x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
#             x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
#             x_embed = torch.cat([x_embed, x_pass], dim=-1)
#             return x_embed.type_as(x) 

# class Memory(nn.Module):
#     def __init__( n, new_dims: int, old_dims: int):
#         super().__init__()

#         n.new_dims = new_dims
#         n.old_dims = old_dims
#         n.new = nn.Linear(new_dims, old_dims)
#         n.old = nn.Linear(old_dims, old_dims)
#         n.activation = nn.Tanh()

#     def forward(n, x: Tensor, y: Tensor) -> Tensor:
#         mix = n.activation(n.new(x) + n.old(y))
#         return mix

#     def initialize_old(n, batch: int, device: torch.device) -> Tensor:
#         old_dims = torch.zeros(batch, n.old_dims).to(device)
#         return old_dims

# class MixtureOfMemories(nn.Module):
#     def __init__(n, dims, head, num_experts: int, expert_size: int, output_size: int):
#         super().__init__()
#         n.mems = nn.ModuleList([nn.Linear(expert_size, output_size) for _ in range(num_experts)])
#         n.gate = nn.Linear(expert_size, num_experts)
#         n.attention = nn.MultiheadAttention(dims, head)
#         n.memory = Memory(dims, dims)
#         n.layernorm = nn.LayerNorm(dims)

#     def forward(n, x: Tensor, y: Tensor) -> Tensor:
#         output, _ = n.attention(x, x, x)
#         y = n.memory(output.mean(dim=0), y)
#         gate_outputs = F.softmax(n.gate(x), dim=1)
#         _outputs = torch.stack([mem(x) for mem in n.mems], dim=1)
#         _output = torch.einsum("ab,abc->ac", gate_outputs, _outputs)
#         output = n.layernorm(output + _output.unsqueeze(0))
#         return output

# class Transformer(nn.Module):
#     def __init__( n, dims: int, head: int, num_experts: int, expert_size: int, num_layers: int):
#         super().__init__()

#         n.layers = nn.ModuleList([MixtureOfMemories( dims, head, num_experts, expert_size ) for _ in range(num_layers)])
#         n.y = torch.zeros(1, dims)

#     def forward(n, x: Tensor) -> Tensor:
#         for layer in n.layers:
#             return layer(x, n.y)
      
# blah = Transformer(dims, head, num_experts, expert_size, num_layers)

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, norm_type, feature=None, norm=False):
        super().__init__()
        
        n.norm = norm
        n.act_fn = get_activation(act)

        # n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True)

        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        n.encoder = nn.Sequential(n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), n.act_fn, 
        nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   
        n.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        
        if n.norm:
            n.ln = get_norm(norm_type, dims) 
        else: 
            n.ln = None 
               
    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1], 10000.0).to(device, dtype)
        if n.norm:
            x = n.ln(x)
        # x = n.EncoderLayer(x)
        return x

class TextEncoder(nn.Module):
    def __init__(n, tokens, dims, head, norm_type, norm=False):
        super().__init__()

        n.norm = norm
        n.embedding = nn.Embedding(tokens, dims)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True)

        if n.norm:
            n.ln = get_norm(norm_type, dims) 
        else: 
            n.ln = None 

    def forward(n, x):
        x = n.embedding(x)
        if n.norm:
            x = n.ln(x)
        return n.EncoderLayer(x)

class CreativeFusionTransformer(nn.Module):
    def __init__(n, tokens, mels, ctx, dims, head, layer, act, norm_type):
        super().__init__()
        n.max_seq_len = ctx

        n.audio_encoder = AudioEncoder(mels, dims, head, act, norm_type, feature = None, norm = False)
        n.text_encoder = TextEncoder(tokens, dims, head, norm_type, norm = False)

        n.blocks = nn.ModuleList([CreativeFusionBlock(dims, head) for _ in range(layer)])

        n.decoder_emb = nn.Embedding(tokens, dims)
        n.decoder_transformer = nn.TransformerDecoderLayer(dims, head, batch_first=True)
        n.decoder_head = nn.Linear(dims, tokens)

    def get_embeddings(n, audio_spec, text_tokens):
        xa = n.audio_encoder(audio_spec)
        x = n.text_encoder(text_tokens)
        audio_emb = torch.mean(xa, dim=1)
        text_emb = torch.mean(x, dim=1)
        return audio_emb, text_emb

    def forward(n, audio, text):

        xa = n.audio_encoder(audio)
        x = n.text_encoder(text)
        
        for block in n.blocks:
            x, xa = block(x, xa)

        decoder_input = n.decoder_emb(text)
        memory = torch.cat([x, xa], dim=1) # Combine fused representations
        output = n.decoder_transformer(decoder_input, memory)
        logits = n.decoder_head(output)
        return logits
    
    def generate(n, audio_spec, start_token_id, end_token_id, max_gen_len=50):
        n.eval()
        with torch.no_grad():
            xa = n.audio_encoder(audio_spec)
            x = torch.zeros(audio_spec.shape[0], 1, n.decoder_emb.embedding_dim, device=audio_spec.device)

            for block in n.blocks:
                x_fused, xa_fused = block(x, xa)
            
            memory = torch.cat([x_fused, xa_fused], dim=1)
            generated_tokens = torch.full((audio_spec.shape[0], 1), start_token_id, dtype=torch.long, device=audio_spec.device)

            for _ in range(max_gen_len):
                decoder_input = n.decoder_emb(generated_tokens)
                output = n.decoder_transformer(decoder_input, memory)
                logits = n.decoder_head(output[:, -1, :])
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
                
                if (next_token_id == end_token_id).all():
                    break
            return generated_tokens

class CreativeFusionBlock(nn.Module):
    def __init__(n, dims: int, head: int, blend: bool = True, modal: bool = True):
        super().__init__()
        n.blend = blend
        n.modal = modal

        n.intra_text = nn.TransformerEncoderLayer(dims, head, batch_first=True)
        n.intra_audio = nn.TransformerEncoderLayer(dims, head, batch_first=True)

        n.cross_modal = AttentionA(dims, head)
   
        if n.modal:
            n.joint_attn = nn.TransformerEncoderLayer(dims * 2, head, batch_first=True)
            
        if n.blend:
            n.register_parameter('blend_weight', nn.Parameter(torch.zeros(1)))

    def forward(n, x, xa, mask=None):
  
        y = x.clone()
        x = n.intra_text(x, src_key_padding_mask=mask)
        xa = n.intra_audio(xa)

        x_fused, xa_fused = n.cross_modal(x, xa, mask=mask) # remove mask
        x = x + x_fused
        xa = xa + xa_fused

        if n.blend:
            alpha = torch.sigmoid(n.blend_weight)
            x = alpha * x + (1 - alpha) * y

        if n.modal:
            xm = n.joint_attn(torch.cat([x, xa], dim=-1))
            x = xm[:, :x.shape[1]]
            xa = xm[:, x.shape[1]:]
        return x, xa

class focus(nn.Module):
    def __init__(n, dims: int, head: int, norm_type="rmsnorm", max_iter: int = 3, threshold: float = 0.5, temp = 1.0):
        super().__init__()

        n.head = head
        n.dims = dims
        n.head_dim = dims // head
        n.win = 0

        n.q   = nn.Sequential(get_norm(norm_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        
        n.register_buffer('freqs_base', compute_freqs_base(dims // head), persistent=False)
        n.rotary = RotaryEmbedding(dims // head, custom_freqs = (36000 / 220.0) * n.freqs_base)  
        n.ln = get_norm(norm_type, dims // head)
        
        n.max_iter = max_iter
        n.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
        n.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        

        n.local = LocalOut(dims, head)   
        
    def update_win(n, win_size=None):
        if win_size is not None:
            n.win_size = win_size
            return win_size
        elif hasattr(n, 'win_size') and n.win_size is not None:
            win_size = n.win_size
            return win_size
        return None

    def _focus(n, x, xa=None, mask=None, win_size=None):
        
        q = n.q(x)
        k, v = n.kv(AorB(xa, x))
        _, _, c, d = q.shape 
        scale = d ** -0.5

        q, k = map(n.rotary.rotate_queries_or_keys, (q, k))

        iteration = 0
        temp = n.temp
        prev_out = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = n.threshold
        curq = q #if curq is None else curq
        
        while iteration < n.max_iter:
            eff_span = min(curq.shape[2], k.shape[2])
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])
            if eff_span == 0: 
                break

            qiter = curq[:, :, :eff_span, :]
            kiter = k[:, :, :eff_span, :]
            viter = v[:, :, :eff_span, :]
            q = n.local.q_hd(qiter)
            k = n.local.k_hd(kiter)
            v = n.local.v_hd(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_iter = calculate_attention(
                n.ln(q), n.ln(k), v,
                mask=iter_mask, temp=temp)

            iter_out = torch.zeros_like(curq)
            iter_out[:, :, :eff_span, :] = attn_iter
            diff = torch.abs(iter_out - prev_out).mean()

            if diff < threshold and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            curq = curq + iter_out
            attn_out = iter_out
            iteration += 1

        return rearrange(attn_out, 'b h c d -> b c (h d)')

    def _slide_win_local(n, x, mask = None) -> Tensor:
        win = n.update_win()
        win_size = win if win is not None else n.head_dim
        span_len = win_size + win_size // n.head

        _, ctx, _ = x.shape
        out = torch.zeros_like(x)
        windows = (ctx + win_size - 1) // win_size

        for q in range(windows):
            qstart = q * win_size
            qend = min(qstart + win_size, ctx)
            qlen = qend - qstart
            if qlen == 0: 
                continue

            kstart = max(0, qend - span_len)
            qwin = x[:, qstart:qend, :]
            kwin = x[:, kstart:qend, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, qstart:qend, kstart:qend]
                elif mask.dim() == 2:
                    win_mask = mask[qstart:qend, kstart:qend]

            attn_out = n._focus(x=qwin, xa=kwin, mask=win_mask, win_size=win_size)
            out[:, qstart:qend, :] = attn_out
        return out

    def forward(n, x, xa = None, mask = None):
        x = n._slide_win_local(x, mask=None)
        xa = n._slide_win_local(xa, mask=None)
        out = n._focus(x, xa, mask=None)
        return n.out(out)

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

class mgate(nn.Module):
    def __init__(n, dims, mem=64, thresh=0.5):
        super().__init__()
        n.mkey = nn.Parameter(torch.randn(mem, dims))
        n.mval = nn.Parameter(torch.randn(mem, 1))
        n.mlp = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        n.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False)
        n.concat = nn.Linear(2,1, device=device, dtype=dtype)
        
    def forward(n, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x = n.concat(torch.cat((torch.matmul(key, n.mval),  n.mlp(x)), dim=-1))
       
        threshold = apply_ste(x, n.threshold)
        return threshold, x

class MiniConnection(nn.Module):
    def __init__(n, dims, expand=2):
        super().__init__()
        n.dims = dims
        n.expand = expand
        n.parallel = nn.ModuleList([nn.Linear(dims, dims) for _ in range(expand)])
        n.network = nn.Linear(dims, expand)
        n.relu = nn.ReLU()
        
    def forward(n, input_features):
        features = [pathway(input_features) for pathway in n.parallel]
        weights = torch.softmax(n.network(input_features), dim=-1)
        unbound_weights = weights.unbind(dim=-1)
        reshaped_weights = [w.unsqueeze(-1) for w in unbound_weights]
        weighted_combined = sum(w * f for w, f in zip(reshaped_weights, features))
        return n.relu(weighted_combined)
        
class skip_layer(nn.Module):
    def __init__(n, dims, head, layer, mini_hc=True, expand=2):
        super().__init__()

        n.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        n.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)
        n.layer = layer
        n.loss = 0
  
        n.layers = nn.ModuleList()
        for q in range(layer):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if q % 2 == 0 else None,
                'mgate': mgate(dims, mem=64),
            }
            if mini_hc:
                layer_dict['mini_hc'] = MiniConnection(dims, expand=expand)
            else:
                layer_dict['mini_hc'] = None

            n.layers.append(nn.ModuleDict(layer_dict))

        n.mgate = mgate(dims, mem=64)
        n.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.SiLU(),
            nn.Linear(128, 3))

        n.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.mlp = nn.Sequential(nn.Linear(dims, dims * 4), nn.SiLU(), nn.Linear(dims * 4, dims))
        n.mlp_ln = nn.LayerNorm(dims)
        
    def update_threshold(n, loss, lr=0.01):
        if loss > n.loss:
            n.mgate.threshold.sub_(lr)
        else:
           n.mgate.threshold.add_(lr)
        n.mgate.threshold.data = torch.clamp(n.mgate.threshold.data, 0.0, 1.0)

    def forward(n, x, xa=None, mask=None): 
        batch, ctx = x.shape[:2]
        ox = x
        work_mem = n.work_mem.expand(batch, -1, -1)
        x1 = x.mean(dim=1)

        policy_logits = n.policy_net(x1)
        policy = F.softmax(policy_logits, dim=-1)
        
        history = []
        q = 0
        while q < n.layer:
            layer = n.layers[q]
            
            scalar, choice = layer['mgate'](x)
            mask_layer = scalar.expand(-1, ctx, -1)
            x2 = torch.zeros_like(x)
            skip = (scalar == 0).squeeze(-1)
            x2[skip] = x[skip]

            px = layer['ln'](x2)  

            if layer['mini_hc'] is not None:
                if layer['adapter'] is not None:
                    adapted_px = layer['adapter'](px)
                else:
                    adapted_px = px
                
                hc_output = layer['mini_hc'](adapted_px)
                gate_val = layer['gate'](px)
                x = x + gate_val * (hc_output * mask_layer)
            else:
                if layer['adapter'] is not None:
                    attn = layer['adapter'](px)
                else:
                    attn = px
                gate_val = layer['gate'](px)
                x = x + gate_val * (attn * mask_layer)

            mem = x.mean(dim=1, keepdim=True)
            mem_val = n.mem_gate(mem)
            work_mem = mem_val * work_mem + (1 - mem_val) * mem
            
            if q < n.layer - 1:
                action = torch.multinomial(policy, 1).squeeze(1).item()
            else:
                action = 0
            distance = 0
            if action == 1: distance = 1
            if action == 2: distance = 2
            if distance > 0:
                i_next = min(q + distance, n.layer - 1)
                jump = n.jump_weights[min(distance-1, 2)]               
                x = x + jump * ox + (1-jump) * work_mem.expand(-1, ctx, -1)
                q = i_next
                history.append(q)
            else:
                q += 1
        
        x3 = n.mlp_gate(x)
        output = n.mlp(n.mlp_ln(x))
        x = x + x3 * output
        n.logs = {'jumps': history}
        return x

# class MiniConnection(nn.Module):
#     def __init__(n, dims, expand=2):
#         super().__init__()
#         n.dims = dims
#         n.expand = expand
#         n.parallel = nn.ModuleList([nn.Linear(dims, dims) for _ in range(expand)])
#         n.network = nn.Linear(dims, expand)
#         n.relu = nn.ReLU()
        
#     def forward(n, input_features):
#         features = [pathway(input_features) for pathway in n.parallel]
#         weights = torch.softmax(n.network(input_features), dim=-1)
#         unbound_weights = weights.unbind(dim=-1)
#         reshaped_weights = [w.unsqueeze(-1) for w in unbound_weights]
#         weighted_combined = sum(w * f for w, f in zip(reshaped_weights, features))
#         return n.relu(weighted_combined)

class Rotaryaa(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("q, k -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

def forward(self, x: torch.Tensor, *, input_pos= None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = (self.cache[:seq_len] if input_pos is None else self.cache[input_pos])
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, head, ctx, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, head, n_rep, ctx, head_dim)
    return x.reshape(batch, head * n_rep, ctx, head_dim)                        

# class HyperBlock(nn.Module):
#     def __init__(n, dims, head, layer, act, norm_type, expand=4):
#         super().__init__()
#         n.main = residual(dims, head, layer, act, norm_type)
#         n.pathways = nn.ModuleList([residual(dims, head, layer, act, norm_type) for _ in range(expand)])
#         n.net = nn.Linear(dims, expand) 
#         n.act = get_activation(act)
        
#     def forward(n, x, xa=None, mask=None):
#         out = n.main(x, xa=xa, mask=mask)
#         eo = [pathway(x, xa=xa, mask=mask) for pathway in n.pathways]
#         wts = torch.softmax(n.net(x), dim=-1)
#         unbound_wts = wts.unbind(dim=-1)
#         reshaped_wts = [w.unsqueeze(-1) for w in unbound_wts]
#         wo = sum(w * f for w, f in zip(reshaped_wts, eo))
#         out = n.act(out + wo)
#         return out



    # In an attention module:
    # rotary = Rotary(dims=head_dim)
    # q, k = rotary.rotate_queries_and_keys(q, k)

    # # Or directly at tensor level, without specifying context:
    # # Apply to any tensor (auto-detects format)
    # x = rotary.apply_rotary(x)

    # # Specialized for attention
    # q, k = rotary.rotate_queries_and_keys(q, k)

    # # Full QKV handling
    # q, k, v = rotary.rotate_qkv(q, k, v)

    # # For token embeddings
    # embeddings = rotary.rotate_token_embeddings(embeddings)

    # # Or with a specific context length:
    # x = rotary.apply_rotary(x, ctx=ctx)  # Shape auto-detected


class Rotarycc(nn.Module):   
    def __init__( self, dims, ctx=1500, freqs_mode='lang', theta=10000, max_freq=10, learned_freq=False, variable_radius=False, learned_radius=False, use_xpos=False, xpos_scale_base=512, interpolate_factor=1.0, cache_if_possible=True, auto_detect_shape=True, debug=False ):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.dims = dims
        self.max_ctx = ctx
        self.variable_radius = variable_radius
        self.cache_if_possible = cache_if_possible
        self.use_xpos = use_xpos
        self.interpolate_factor = interpolate_factor
        self.auto_detect_shape = auto_detect_shape
        
        if freqs_mode == 'lang':
            freqs = 1.0 / (theta ** (torch.arange(0, dims, 2).float() / dims))
        elif freqs_mode == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dims // 2) * pi
        elif freqs_mode == 'constant':
            freqs = torch.ones(dims // 2).float()
        else:
            raise ValueError(f"Unknown freqs_mode: {freqs_mode}")
            
        self.inv_freq = nn.Parameter(freqs, requires_grad=learned_freq)
        self.freqs_mode = freqs_mode
        
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )
            
        self.bias = nn.Parameter(torch.zeros(ctx, dims // 2))
        
        if use_xpos:
            scale = (torch.arange(0, dims, 2) + 0.4 * dims) / (1.4 * dims)
            self.scale_base = xpos_scale_base
            self.register_buffer('scale', scale, persistent=False)
            

        if use_xpos:
            self.register_buffer('cached_scales', torch.zeros(ctx, dims), persistent=False)
            self.cached_scales_ctx = 0
            
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.register_buffer('cached_freqs', torch.zeros(ctx, dims // 2), persistent=False)
        self.cached_freqs_ctx = 0
        
        self.ctx_cache = {}
        self.max_cache_entries = 16
        
        self.register_buffer('param_version', torch.tensor(0), persistent=False)
    
    def invalidate_cache(self):
        self.cached_freqs_ctx = 0
        self.ctx_cache.clear()
        self.param_version += 1

    @property
    def device(self):
        return self.dummy.device
    
    def get_seq_pos(self, ctx, device, dtype, offset=0):
        return (torch.arange(ctx, device=device, dtype=dtype) + offset) / self.interpolate_factor
    
    def get_scale(self, t, ctx=None, offset=0):
        assert self.use_xpos
        should_cache = (self.cache_if_possible and
            exists(ctx) and (offset + ctx) <= self.max_ctx)
        
        if (should_cache and exists(self.cached_scales) and
            (ctx + offset) <= self.cached_scales_ctx):
            return self.cached_scales[offset:(offset + ctx)]
            
        power = (t - len(t) // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = repeat(scale, 'n d -> n (d r)', r=2)
        
        if should_cache and offset == 0:
            self.cached_scales[:ctx] = scale.detach()
            self.cached_scales_ctx = ctx
            
        return scale
    
    # @autocast('cuda', enabled=False)
    def forward(self, x, offset=0):
        if isinstance(x, int):
            t = torch.arange(x, device=self.device, dtype=torch.float32)
            ctx = x
        else:
            t = x.float().to(self.device)
            ctx = t.shape[0] if hasattr(t, 'shape') else t
        
        skip_cache = self.inv_freq.requires_grad or (
            self.variable_radius and self.radius.requires_grad)
        
        if not skip_cache and self.cache_if_possible:
            if ctx <= self.max_ctx and offset == 0:
                if ctx <= self.cached_freqs_ctx:
                    return self.cached_freqs[:ctx].unsqueeze(0)
            
            cache_key = (ctx, offset, self.param_version.item())
            if cache_key in self.ctx_cache:
                return self.ctx_cache[cache_key]

        if self.freqs_mode == 'lang':
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        else:
            freqs = einsum('..., f -> ... f', t.type(self.inv_freq.dtype), self.inv_freq)
        freqs = freqs + self.bias[offset:offset+freqs.shape[0]]
        
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)
        
        if not skip_cache and self.cache_if_possible:
            if ctx <= self.max_ctx and offset == 0:
                self.cached_freqs[:ctx] = freqs.squeeze(0)
                self.cached_freqs_ctx = max(self.cached_freqs_ctx, ctx)
            else:
                if len(self.ctx_cache) >= self.max_cache_entries:
                    self.ctx_cache.pop(next(iter(self.ctx_cache)))
                self.ctx_cache[cache_key] = freqs.detach()
            
        if self.debug and self._counter < 1:
            print(f'ROTARY -- freqs: {freqs.shape}, t: {t.shape if hasattr(t, "shape") else None}')
            self._counter += 1
        return freqs
    
    def _reshape_for_multihead(self, freqs, head, head_dim=None):
        head_dim = head_dim or self.dims // head
        ctx = freqs.shape[1]
        complex_per_head = head_dim // 2
        
        if complex_per_head * head > freqs.shape[2]:
            freqs = freqs[:, :, :complex_per_head * head]
        elif complex_per_head * head < freqs.shape[2]:
            padding = torch.zeros(
                (freqs.shape[0], ctx, complex_per_head * head - freqs.shape[2]),
                device=freqs.device,
                dtype=freqs.dtype
            )
            freqs = torch.cat([freqs, padding], dim=2)
            
        return freqs.view(freqs.shape[0], ctx, head, complex_per_head)
    
    def _detect_tensor_format(self, tensor):
        shape = tensor.shape
        
        if len(shape) == 3:
            return {
                'format': 'sequence',
                'batch': shape[0],
                'ctx': shape[1], 
                'dim': shape[2],
                'is_multi_head': False
            }
        elif len(shape) == 4:
            return {
                'format': 'multihead',
                'batch': shape[0],
                'head': shape[1],
                'ctx': shape[2],
                'head_dim': shape[3],
                'is_multi_head': True
            }
        else:
            raise ValueError(f"Unsupported tensor shape: {shape}")
    
    def apply_rotary(self, x, freqs=None, seq_dim=-2, start_index=0, scale=1.0):
        """
        Args:
            x: Tensor to apply rotary embeddings to
            freqs: Optional pre-computed frequency tensor
            seq_dim: Dimension containing sequence positions
            start_index: Starting index for partial rotation
            scale: Scaling factor for rotations 
        """
        format_info = self._detect_tensor_format(x) if self.auto_detect_shape else None
        
        if format_info:
            if format_info['is_multi_head']:
                batch, head, ctx, head_dim = (format_info['batch'], format_info['head'], 
                    format_info['ctx'], format_info['head_dim'])
                
                if freqs is None:
                    freqs = self.forward(ctx)
                    freqs = self._reshape_for_multihead(freqs, head, head_dim)
                    freqs = freqs.permute(0, 2, 1, 3)
                
                x1 = x[..., :freqs.shape[-1]*2]
                x2 = x[..., freqs.shape[-1]*2:]
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs * scale
                x1 = torch.view_as_real(x1).flatten(-2)
                return torch.cat([x1.type_as(x), x2], dim=-1)
            else:
                ctx = format_info['ctx']
                
                if freqs is None:
                    freqs = self.forward(ctx)
                
                return self._apply_rotary_to_sequence(x, freqs, start_index, scale)
        else:
            if freqs is None:
                if seq_dim < 0:
                    seq_dim = len(x.shape) + seq_dim
                ctx = x.shape[seq_dim]
                freqs = self.forward(ctx)
            
            return self._apply_rotary_to_sequence(x, freqs, start_index, scale)
    
    def _apply_rotary_to_sequence(self, x, freqs, start_index=0, scale=1.0):
        rot_dim = freqs.shape[-1] * 2
        end_index = start_index + rot_dim
        
        assert rot_dim <= x.shape[-1], f'Feature dimension {x.shape[-1]} is too small for rotation size {rot_dim}'
        
        t_left = x[..., :start_index]
        t_middle = x[..., start_index:end_index]
        t_right = x[..., end_index:]
        
        if torch.is_complex(freqs):
            cos = freqs.real
            sin = freqs.imag
        else:
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        
        t_transformed = (t_middle * cos * scale) + (rotate_half(t_middle) * sin * scale)
        return torch.cat((t_left, t_transformed, t_right), dim=-1)
    
    def rotate_queries_and_keys(self, q, k, scale=1.0):
        q_info = self._detect_tensor_format(q) if self.auto_detect_shape else None
        k_info = self._detect_tensor_format(k) if self.auto_detect_shape else None
        q_len = q_info['ctx'] if q_info else q.shape[-2]
        k_len = k_info['ctx'] if k_info else k.shape[-2]
        q_freqs = self.forward(q_len)
        
        if q_len != k_len:
            k_freqs = self.forward(k_len)
        else:
            k_freqs = q_freqs
            
        if self.use_xpos:
            q_seq = self.get_seq_pos(q_len, device=q.device, dtype=q.dtype)
            k_seq = self.get_seq_pos(k_len, device=k.device, dtype=k.dtype)
            q_scale = self.get_scale(q_seq).to(q.dtype) * scale
            k_scale = self.get_scale(k_seq).to(k.dtype) ** -1
        else:
            q_scale = scale
            k_scale = 1.0
        
        rotated_q = self.apply_rotary(q, q_freqs, scale=q_scale)
        rotated_k = self.apply_rotary(k, k_freqs, scale=k_scale)
        return rotated_q, rotated_k
    
    def rotate_qkv(self, q, k, v, scale=1.0):
        rotated_q, rotated_k = self.rotate_queries_and_keys(q, k, scale)
        return rotated_q, rotated_k, v
    
    def rotate_token_embeddings(self, token_emb, scale=1.0):
        ctx = token_emb.shape[1]
        freqs = self.forward(ctx)
        return self.apply_rotary(token_emb, freqs, scale=scale)
    


# class RotaryLite(nn.Module):
#     def __init__(self, dims, max_ctx=1500, learned_freq=True, variable_radius=True, learned_radius=True):
#         super().__init__()
#         self.dims = dims
#         self.variable_radius = variable_radius
#         self.inv_freq = nn.Parameter(
#             1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
#             requires_grad=learned_freq
#         )
        
#         if variable_radius:
#             self.radius = nn.Parameter(
#                 torch.ones(dims // 2),
#                 requires_grad=learned_radius
#             )
        
#         self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
#     def forward(self, positions):
#         if isinstance(positions, int):
#             t = torch.arange(positions, device=self.inv_freq.device).float()
#         else:
#             t = positions.float().to(self.inv_freq.device)
#         freqs = torch.einsum('i,j->ij', t, self.inv_freq)
#         freqs = freqs + self.bias[:freqs.shape[0]]
#         if self.variable_radius:
#             radius = F.softplus(self.radius)
#             freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
#         else:
#             freqs = torch.polar(torch.ones_like(freqs), freqs)
#         return freqs
    
#     def _reshape_for_multihead(self, freqs, head, head_dim):
#         ctx = freqs.shape[0]
#         complex_per_head = head_dim // 2
#         if complex_per_head * head > freqs.shape[1]:
#             freqs = freqs[:, :complex_per_head * head]
#         elif complex_per_head * head < freqs.shape[1]:
#             padding = torch.zeros(
#                 (ctx, complex_per_head * head - freqs.shape[1]), 
#                 device=freqs.device, 
#                 dtype=freqs.dtype
#             )
#             freqs = torch.cat([freqs, padding], dim=1)
#         freqs = freqs.view(ctx, head, complex_per_head)
#         return freqs.permute(2, 1, 0, 2).unsqueeze(0)

#     @staticmethod
#     def apply_rotary(x, freqs):
#         multihead_format = len(freqs.shape) == 4
        
#         if multihead_format:
#             x1 = x[..., :freqs.shape[-1]*2]
#             x2 = x[..., freqs.shape[-1]*2:]
#             x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
#             x1 = torch.view_as_complex(x1)
#             x1 = x1 * freqs
#             x1 = torch.view_as_real(x1).flatten(-2)
#             return torch.cat([x1.type_as(x), x2], dim=-1)
#         else:
#             x1 = x[..., :freqs.shape[-1]*2]
#             x2 = x[..., freqs.shape[-1]*2:]
#             x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
#             x1 = torch.view_as_complex(x1)
#             x1 = x1 * freqs
#             x1 = torch.view_as_real(x1).flatten(-2)
#             return torch.cat([x1.type_as(x), x2], dim=-1)

class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, 
                 use_freq_bands=False, speech_enhanced=False,
                 variable_radius=False, learned_radius=True, init_radius=1.0):
        super().__init__()
        self.dims = dims
        self.use_freq_bands = use_freq_bands
        self.variable_radius = variable_radius
        
        # Configure frequency parameters
        if not use_freq_bands:
            # Original implementation
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
                requires_grad=learned_freq
            )
            self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Global radius parameter (if variable)
            if variable_radius:
                self.radius = nn.Parameter(
                    torch.ones(dims // 2) * init_radius,
                    requires_grad=learned_radius
                )
        else:
            # FrequencyBand implementation
            band_size = dims // 6  # Each band gets 1/3 of dims (x2 for complex numbers)
            
            # Low frequencies (0-500Hz range in speech)
            self.low_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Mid frequencies (500-2000Hz in speech)
            self.mid_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(band_size, 2*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # High frequencies (>2000Hz in speech)
            self.high_freq_audio = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Text-specific high frequencies
            self.high_freq_text = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Frequency-specific biases
            if speech_enhanced:
                self.low_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.mid_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.high_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
            else:
                self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Band-specific radius parameters (if variable)
            if variable_radius:
                self.low_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.mid_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_audio = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_text = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                
        self.speech_enhanced = speech_enhanced and use_freq_bands

    def forward(self, positions, domain="audio", snr_estimate=None):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.get_device()).float()
        else:
            t = positions.float().to(self.get_device())
        
        if not self.use_freq_bands:
            # Original implementation
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            freqs = freqs + self.bias[:freqs.shape[0]]
            
            if self.variable_radius:
                # Apply learnable radius instead of fixed radius=1
                radius = F.softplus(self.radius)  # Ensure radius is positive
                freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
            else:
                # Original fixed radius
                freqs = torch.polar(torch.ones_like(freqs), freqs)
        else:
            # FrequencyBand implementation
            low = torch.einsum('i,j->ij', t, self.low_freq)
            mid = torch.einsum('i,j->ij', t, self.mid_freq)
            
            # Domain-specific high frequencies
            if domain == "audio":
                high = torch.einsum('i,j->ij', t, self.high_freq_audio)
            else:
                high = torch.einsum('i,j->ij', t, self.high_freq_text)
            
            # Apply bias
            if self.speech_enhanced:
                low = low + self.low_bias[:low.shape[0]]
                mid = mid + self.mid_bias[:mid.shape[0]]
                high = high + self.high_bias[:high.shape[0]]
            else:
                # Create full bias-adjusted frequencies before applying radius
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = freqs + self.bias[:freqs.shape[0]]
                low, mid, high = torch.split(freqs, freqs.shape[1]//3, dim=1)
            
            # Apply variable radius if enabled
            if self.variable_radius:
                # Get appropriate radius for each band
                low_radius = F.softplus(self.low_radius)
                mid_radius = F.softplus(self.mid_radius)
                
                if domain == "audio":
                    high_radius = F.softplus(self.high_radius_audio)
                else:
                    high_radius = F.softplus(self.high_radius_text)
                
                # Adjust radius based on SNR if provided (audio mode only)
                if snr_estimate is not None and domain == "audio":
                    # Convert SNR to a scaling factor (lower SNR = smaller high freq radius)
                    snr_factor = torch.sigmoid((snr_estimate - 5) / 5)  # Maps to 0-1
                    
                    # Apply progressively stronger scaling to higher frequencies
                    # (high frequencies most affected by noise)
                    low_radius = low_radius  # Low frequencies mostly preserved
                    mid_radius = mid_radius * (0.5 + 0.5 * snr_factor)  # Partial scaling
                    high_radius = high_radius * snr_factor  # Strongest scaling
                
                # Create complex numbers with variable radius for each band
                low_complex = torch.polar(low_radius.unsqueeze(0).expand_as(low), low)
                mid_complex = torch.polar(mid_radius.unsqueeze(0).expand_as(mid), mid)
                high_complex = torch.polar(high_radius.unsqueeze(0).expand_as(high), high)
                
                # Combine all bands
                freqs = torch.cat([low_complex, mid_complex, high_complex], dim=-1)
            else:
                # Use fixed radius=1 (original behavior)
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = torch.polar(torch.ones_like(freqs), freqs)
                
        return freqs
    
    def get_device(self):
        """Helper to get device from any parameter"""
        if hasattr(self, 'inv_freq'):
            return self.inv_freq.device
        return self.low_freq.device
        
    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)



# # Assume we have pre-processed features for two audio modalities and text
# # This would typically be a custom Dataset and DataLoader in a real project
# # For this example, we'll use random tensors.
# batch_size = 16
# audio_feature_1_dim = 128
# audio_feature_2_dim = 64
# text_feature_dim = 256
# embedding_dim = 512

# # --- Step 1: Define the Encoders ---

# class AudioEncoder1(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.encoder(x)

# class AudioEncoder2(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.encoder(x)

# class TextEncoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.encoder(x)

# # --- Step 2: Define the Contrastive Loss Function ---

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, audio_embeddings, text_embeddings):
#         # Normalize embeddings to be on the unit sphere
#         audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
#         text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

#         # Calculate cosine similarity matrix
#         # (A @ T.T) where A is audio embeddings and T is text embeddings
#         similarity_matrix = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature

#         # Create labels for the diagonal (correct pairs)
#         labels = torch.arange(similarity_matrix.size(0)).long().to(similarity_matrix.device)

#         # Calculate loss using cross-entropy, as is common with contrastive losses (e.g., NT-Xent)
#         loss_audio = F.cross_entropy(similarity_matrix, labels)
#         loss_text = F.cross_entropy(similarity_matrix.T, labels)

#         # The total loss is the average of the two
#         return (loss_audio + loss_text) / 2

# # --- Step 3: Set up the model and training loop ---

# # Instantiate models
# audio_encoder_1 = AudioEncoder1(audio_feature_1_dim, embedding_dim)
# audio_encoder_2 = AudioEncoder2(audio_feature_2_dim, embedding_dim)
# text_encoder = TextEncoder(text_feature_dim, embedding_dim)

# # Combine audio encoders into a single module for convenience
# class MultiModalEncoder(nn.Module):
#     def __init__(self, audio_encoder_1, audio_encoder_2):
#         super().__init__()
#         self.audio_encoder_1 = audio_encoder_1
#         self.audio_encoder_2 = audio_encoder_2
#         # A simple linear layer to combine the features
#         self.fusion_head = nn.Linear(embedding_dim * 2, embedding_dim)
    
#     def forward(self, audio_1, audio_2):
#         emb_1 = self.audio_encoder_1(audio_1)
#         emb_2 = self.audio_encoder_2(audio_2)
#         fused_embeddings = torch.cat((emb_1, emb_2), dim=1)
#         return self.fusion_head(fused_embeddings)

# multi_modal_audio_encoder = MultiModalEncoder(audio_encoder_1, audio_encoder_2)

# # Instantiate loss and optimizer
# loss_fn = ContrastiveLoss()
# optimizer = torch.optim.Adam(
#     list(multi_modal_audio_encoder.parameters()) + list(text_encoder.parameters()),
#     lr=1e-4
# )

# # --- Dummy training loop ---

# print("Starting training...")
# for epoch in range(50):
#     # Generate dummy data for a single batch
#     # In a real scenario, this would come from your DataLoader
#     audio_features_1 = torch.randn(batch_size, audio_feature_1_dim)
#     audio_features_2 = torch.randn(batch_size, audio_feature_2_dim)
#     text_features = torch.randn(batch_size, text_feature_dim)
    
#     # Randomly shuffle negative text samples
#     shuffled_text_features = text_features[torch.randperm(batch_size)]

#     # Zero gradients
#     optimizer.zero_grad()

#     # Get embeddings for all modalities
#     audio_embeddings = multi_modal_audio_encoder(audio_features_1, audio_features_2)
#     text_embeddings = text_encoder(text_features)
    
#     # Calculate loss on the aligned (correct) pairs
#     loss = loss_fn(audio_embeddings, text_embeddings)

#     # Backpropagate and update weights
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# print("\nTraining finished.")

# # --- Step 4: Example inference and evaluation ---
# print("--- Inference Example ---")
# with torch.no_grad():
#     # Example 1: a matching audio and text pair
#     true_audio_emb = multi_modal_audio_encoder(audio_features_1[0].unsqueeze(0), audio_features_2[0].unsqueeze(0))
#     true_text_emb = text_encoder(text_features[0].unsqueeze(0))

#     # Example 2: a mismatched audio and text pair
#     mismatched_text_emb = text_encoder(shuffled_text_features[0].unsqueeze(0))

#     # Calculate cosine similarity for evaluation
#     cosine_sim = nn.CosineSimilarity(dim=1)
    
#     match_similarity = cosine_sim(true_audio_emb, true_text_emb)
#     mismatch_similarity = cosine_sim(true_audio_emb, mismatched_text_emb)

#     print(f"\nSimilarity between CORRECT audio and text: {match_similarity.item():.4f}")
#     print(f"Similarity between MISMATCHED audio and text: {mismatch_similarity.item():.4f}")

#     # After successful training, we expect match_similarity to be significantly higher
#     # than mismatch_similarity.


# class PairwiseDistanceLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin
#         self.pairwise_distance = nn.PairwiseDistance(p=2, keepdim=False)

#     def forward(self, embedding_a, embedding_t, is_same_pair):
#         """
#         Calculates the margin-based contrastive loss using PairwiseDistance.

#         Args:
#             embedding_a (torch.Tensor): Embeddings for the audio modality.
#             embedding_t (torch.Tensor): Embeddings for the text modality.
#             is_same_pair (torch.Tensor): A boolean tensor where True indicates
#                                          a positive pair and False a negative.
#         Returns:
#             torch.Tensor: The calculated loss value.
#         """
#         # Calculate the Euclidean distance for each pair
#         distances = self.pairwise_distance(embedding_a, embedding_t)
        
#         # Determine the loss for positive and negative pairs
#         loss_positive = (1 - is_same_pair.float()) * distances.pow(2)
#         loss_negative = is_same_pair.float() * F.relu(self.margin - distances).pow(2)
        
#         # Sum both losses and take the mean
#         loss = torch.mean(loss_positive + loss_negative)
#         return loss

# # --- Example of how to integrate with the previous code ---
# # You can replace the ContrastiveLoss with this new module.

# # Instantiate the new loss module
# pdist_loss_fn = PairwiseDistanceLoss(margin=2.0)

# # --- Dummy training loop (abbreviated) ---
# # Assuming the encoders (MultiModalEncoder and TextEncoder) are already defined and instantiated.
# # ... (from the previous example) ...

# # For demonstration, we'll create a batch of half positive and half negative pairs
# batch_size = 16
# half_batch = batch_size // 2

# # Audio/text features (dummy data)
# audio_features_1 = torch.randn(batch_size, audio_feature_1_dim)
# audio_features_2 = torch.randn(batch_size, audio_feature_2_dim)
# text_features = torch.randn(batch_size, text_feature_dim)

# # Create positive and negative pairs for the batch
# # Positive pairs: audio[0:half] with text[0:half]
# # Negative pairs: audio[half:batch] with text[0:half] (mismatched)
# positive_audio_1 = audio_features_1[:half_batch]
# positive_audio_2 = audio_features_2[:half_batch]
# positive_text = text_features[:half_batch]

# negative_audio_1 = audio_features_1[half_batch:]
# negative_audio_2 = audio_features_2[half_batch:]
# negative_text = text_features[:half_batch] # Mismatched text

# # Concatenate for full batch
# full_audio_1 = torch.cat([positive_audio_1, negative_audio_1], dim=0)
# full_audio_2 = torch.cat([positive_audio_2, negative_audio_2], dim=0)
# full_text = torch.cat([positive_text, negative_text], dim=0)

# # Labels for the batch (0 for positive, 1 for negative)
# is_same_pair = torch.cat([torch.zeros(half_batch, dtype=torch.bool), torch.ones(half_batch, dtype=torch.bool)], dim=0)

# # Training step
# optimizer.zero_grad()
# audio_embeddings = multi_modal_audio_encoder(full_audio_1, full_audio_2)
# text_embeddings = text_encoder(full_text)

# # Calculate the loss using PairwiseDistanceLoss
# loss = pdist_loss_fn(audio_embeddings, text_embeddings, is_same_pair)

# loss.backward()
# optimizer.step()

# print(f"Loss with PairwiseDistance: {loss.item():.4f}")

class taylor_rotary(nn.Module):#Taylor series expansion of sine and cosine experimental
    def __init__(n, dims, head, taylor_order=2):
        super().__init__()
        n.dims = dims
        n.head = head
        n.head_dim = dims // head
        n.variable_radius = False
        n.taylor_order = taylor_order

        n.radius = nn.Parameter(torch.ones(n.head_dim // 2), requires_grad=True)
        n.theta = nn.Parameter((torch.tensor(10000, device=device, dtype=dtype)), requires_grad=True)  
        n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(n):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def taylor_sine(n, x, order=2):
        result = torch.zeros_like(x)
        for i in range(order + 1):
            if i % 2 == 1:  
                term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
                if (i // 2) % 2 == 1: 
                    result -= term
                else:
                    result += term
        return result

    def taylor_cosine(n, x, order=2):
        result = torch.zeros_like(x)
        for i in range(order + 1):
            if i % 2 == 0:  
                term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
                if (i // 2) % 2 == 1: 
                    result -= term
                else:
                    result += term
        return result

    def rotate_half(n, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(n, x) -> torch.Tensor:

        positions = (torch.arange(0, x.shape[2], device=device))
        freqs = (n.theta / 220.0) * n.freqs_base
        freqs = positions[:, None] * freqs 

        freqs = (freqs + torch.pi) % (2 * torch.pi) - torch.pi

        with torch.autocast(device_type="cuda", enabled=False):
            cos = n.taylor_cosine(freqs, order=n.taylor_order)
            sin = n.taylor_sine(freqs, order=n.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (n.rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x) 



class vtaylor_rotary(nn.Module):#Taylor series expansion of sine and cosine experimental
    def __init__(n, dims, head, taylor_order=2):
        super().__init__()
        n.dims = dims
        n.head = head
        n.head_dim = dims // head
        n.variable_radius = False
        n.taylor_order = taylor_order

        n.radius = nn.Parameter(torch.ones(n.head_dim // 2), requires_grad=True)
        n.theta = nn.Parameter((torch.tensor(10000, device=device, dtype=dtype)), requires_grad=True)  
        n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(n):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 


    def taylor_sine(n, x, order=2):
        og_shape = x.shape
        x = x.flatten(0, -2)
        exponents = torch.arange(1, order + 1, 2, device=x.device, dtype=torch.float32)
        x_powers = x.unsqueeze(-1) ** exponents
        factorials = torch.exp(torch.lgamma(exponents + 1))
        signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
        terms = signs * x_powers / factorials
        result = terms.sum(dim=-1)
        return result.view(og_shape)

    def taylor_cosine(n, x, order=2):
        og_shape = x.shape
        x = x.flatten(0, -2)
        exponents = torch.arange(0, order + 1, 2, device=x.device, dtype=torch.float32)
        x_powers = x.unsqueeze(-1) ** exponents
        factorials = torch.exp(torch.lgamma(exponents + 1))
        signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
        terms = signs * x_powers / factorials
        result = terms.sum(dim=-1)
        return result.view(og_shape)

    def rotate_half(n, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(n, x) -> torch.Tensor:

        positions = (torch.arange(0, x.shape[2], device=device))
        freqs = (n.theta / 220.0) * n.freqs_base
        freqs = positions[:, None] * freqs 

        freqs = (freqs + torch.pi) % (2 * torch.pi) - torch.pi

        with torch.autocast(device_type="cuda", enabled=False):
            cos = n.taylor_cosine(freqs, order=n.taylor_order)
            sin = n.taylor_sine(freqs, order=n.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (n.rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x) 
