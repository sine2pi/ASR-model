
import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Dict
import gzip
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizerFast
from typing import Union, List, Any
import evaluate
import transformers
from dataclasses import dataclass
from itertools import chain

# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0")
dtype = torch.float32

torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    decoder_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    encoder_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str 

def visualize_attention_weights(attn_weights):
    import seaborn as sns
    batch, heads, seq_len, _ = attn_weights.shape
    plt.figure(figsize=(12, 4))
    for h in range(min(4, heads)):
        plt.subplot(1, min(4, heads), h+1)
        sns.heatmap(attn_weights[0, h].detach().cpu().numpy())
        plt.title(f'Head {h}')
    plt.suptitle("Attention Weights")
    plt.show()

def visualize_rotary_angles(rotary, seq_len):
    freqs = rotary.inv_freq.detach().cpu().numpy()
    t = np.arange(seq_len)
    angles = np.outer(t, freqs)
    plt.figure(figsize=(10, 6))
    for i in range(min(4, angles.shape[1])):
        plt.plot(angles[:, i], label=f'Freq {i}')
    plt.title("Rotary Angles per Position")
    plt.xlabel("Position")
    plt.ylabel("Angle (radians)")
    plt.legend()
    plt.show()

def visualize_rotary_effects(x, rotary):
    seq_len = x.shape[1]
    freqs_cis = rotary(seq_len)
    x_rot = rotary.apply_rotary(x, freqs_cis)
    idx = 0
    dims_to_plot = [0, 1, 2, 3]
    plt.figure(figsize=(10, 6))
    for d in dims_to_plot:
        plt.plot(x[idx, :, d].detach().cpu().numpy(), label=f'Orig dim {d}')
        plt.plot(x_rot[idx, :, d].detach().cpu().numpy(), '--', label=f'Rotary dim {d}')
    plt.title("Effect of Rotary on Embedding Dimensions")
    plt.xlabel("Sequence Position")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.show()
    

def plot_betweenness(be, title="Betweenness"):
    """
    Plots betweenness for a batch of sequences.
    Args:
        be: Tensor of shape (batch, seq_len)
    """
    be = be.detach().cpu().numpy()
    plt.figure(figsize=(12, 3))
    for i in range(min(4, be.shape[0])):
        plt.plot(be[i], label=f"Sample {i}")
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Betweenness")
    plt.legend()
    plt.show()

def plot_waveform_and_spectrogram(waveform, spectrogram, sample_idx=0, sr=16000, title="Waveform and Spectrogram"):
    """
    Plots the waveform and spectrogram for a single sample.
    Args:
        waveform: Tensor of shape (batch, 1, n_samples) or (batch, n_samples)
        spectrogram: Tensor of shape (batch, seq_len, n_mels) or (batch, n_mels, seq_len)
        sample_idx: which sample in the batch to plot
        sr: sample rate for x-axis scaling (default 16kHz)
    """
    wf = waveform[sample_idx].detach().cpu().numpy()
    if wf.ndim > 1:
        wf = wf.squeeze()
    t = np.arange(len(wf)) / sr

    spec = spectrogram[sample_idx].detach().cpu().numpy()
    if spec.shape[0] < spec.shape[1]:
        spec = spec.T

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    axs[0].plot(t, wf, color="tab:blue")
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    axs[1].imshow(spec.T, aspect="auto", origin="lower", cmap="magma")
    axs[1].set_title("Spectrogram")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Mel Bin")
    plt.tight_layout()
    plt.show()

def plot_betweenness_overlay(be, x, sample_idx=0, title="Betweenness Overlay"):
    """
    Overlay betweenness with spectrogram and energy for a single sample.
    Args:
        be: Tensor of shape (batch, seq_len)
        x: Tensor of shape (batch, seq_len, n_mels) or (batch, n_mels, seq_len)
        sample_idx: which sample in the batch to plot
    """
    import matplotlib.pyplot as plt

    be = be[sample_idx].detach().cpu().numpy()
    if x.shape[1] != be.shape[0] and x.shape[-1] == be.shape[0]:
        x = x.permute(0, 2, 1)
    spec = x[sample_idx].detach().cpu().numpy()
    energy = spec.mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.set_title(title)
    ax1.set_xlabel("Sequence Position")
    ax1.set_ylabel("Betweenness", color="tab:red")
    ax1.plot(be, color="tab:red", label="Betweenness")
    ax1.tick_params(axis='y', labelcolor="tab:red")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy", color="tab:blue")
    ax2.plot(energy, color="tab:blue", alpha=0.5, label="Energy")
    ax2.tick_params(axis='y', labelcolor="tab:blue")
    ax2.legend(loc="upper right")

    plt.show()

    plt.figure(figsize=(14, 3))
    plt.imshow(spec.T, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Spectrogram (dB)")
    plt.title("Input Spectrogram")
    plt.xlabel("Sequence Position")
    plt.ylabel("Mel Bin")
    plt.show()

class BetweennessModule(nn.Module):
    def __init__(self, dim, adjustment_scale=1.0, window_size=10):
        super().__init__()
        self.dim = dim
        self.adjustment_scale = adjustment_scale
        self.content_proj = nn.Linear(dim, dim)
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def compute_betweenness(self, x):
        batch, seq_len, dim = x.shape
        content = self.norm(self.content_proj(self.dropout(x)))
        device = x.device
        window = self.window_size

        betweenness = torch.zeros(batch, seq_len, device=device)

        for offset in range(1, window + 1):
            n_indices = seq_len - 2 * offset
            if n_indices <= 0:
                continue
            i = torch.arange(n_indices, device=device)
            j = i + offset
            k = i + 2 * offset

            c_i = content[:, i, :]
            c_j = content[:, j, :]
            c_k = content[:, k, :]

            def cos_dist(a, b):
                a = F.normalize(a, dim=-1)
                b = F.normalize(b, dim=-1)
                return 1 - (a * b).sum(dim=-1)

            direct = cos_dist(c_i, c_k)
            path = cos_dist(c_i, c_j) + cos_dist(c_j, c_k)
            safe_direct = torch.clamp(direct, min=1e-3)
            between_score = 1.0 - (path - direct) / safe_direct
            betweenness[:, j] += between_score

        betweenness = betweenness / max(window, 1)
        betweenness = betweenness - betweenness.mean(dim=1, keepdim=True)
        std = betweenness.std(dim=1, keepdim=True) + 1e-6
        betweenness = betweenness / std

        betweenness = self.betweenness_gate * self.adjustment_scale * betweenness
        betweenness = torch.clamp(betweenness, -2.0, 2.0)

        return betweenness

def apply_to_rope(rope_func, x, positions, betweenness_module):
    adjustments = betweenness_module.get_position_adjustments(x)
    adjusted_positions = positions + adjustments
    return rope_func(x, adjusted_positions)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.RMSNorm):       
    def forward(self, x: Tensor) -> Tensor:
        """Preserve the input dtype throughout the normalization"""
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        eps = self.eps if self.eps is not None else torch.finfo(x_float.dtype).eps
        x_normalized = x_float * torch.rsqrt(variance + eps)
        if self.weight is not None:
            return (x_normalized * self.weight).type(x.dtype)
        return x_normalized.type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

class ParameterCycler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.current_idx = 0

    def toggle_requires_grad(self):
        for i, param in enumerate(self.parameters):
            param.requires_grad = i == self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.parameters)

def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=4096, learned_freq=True):
        super().__init__()
        self.dim = dim
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=learned_freq
        )
        self.bias = nn.Parameter(torch.zeros(max_seq_len, dim // 2))  

    def forward(self, positions):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.inv_freq.device).float()
        else:
            t = positions.float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    @staticmethod
    def apply_rotary(x, freqs_cis):
        x1 = x[..., :freqs_cis.shape[-1]*2]
        x2 = x[..., freqs_cis.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs_cis
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
    
class Multihead(nn.Module):
    blend = False
    cos = False
    mag = False

    def __init__(self, dims: int, head: int):
        super().__init__()
        self.dims = dims
        self.head = head
        head_dim = dims // head
        self.head_dim = head_dim
        self.dropout = 0.1

        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)

        self.use_betweenness = False  

        if self.use_betweenness:
            self.betweenness = BetweennessModule(dim=head_dim, window_size=10)

        self.rotary = Rotary(dim=head_dim, learned_freq=True)

        if Multihead.blend:
            self.factor = nn.Parameter(torch.tensor(0.5, **tox))

    def compute_cosine_attention(self, q: Tensor, k: Tensor, v: Tensor, mask):
        ctx = q.shape[1]
        qn = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        kn = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk = torch.matmul(qn, kn.transpose(-1, -2))

        if Multihead.mag:
            qm = torch.norm(q, dim=-1, keepdim=True)
            km = torch.norm(k, dim=-1, keepdim=True)
            ms = (qm * km.transpose(-1, -2)) ** 0.5
            ms = torch.clamp(ms, min=1e-8)
            qk = qk * ms

        if mask is not None:
            qk = qk + mask[:ctx, :ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        w = F.dropout(w, p=self.dropout, training=self.training)
        out = torch.matmul(w, v)
        return out, qk

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask = None, kv_cache = None):
        q = self.q(x)
        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]
        out, qk = self._forward(q, k, v, mask)
        return self.o(out), qk

    def _forward(self, q: Tensor, k: Tensor, v: Tensor, mask = None):
        ctx_q = q.shape[1]
        ctx_k = k.shape[1]
        ctx = q.shape[1]
        dims = self.dims
        scale = (dims // self.head) ** -0.25

        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        if q.shape[2] == k.shape[2]:
            freqs_cis = self.rotary(ctx_q)
            q = self.rotary.apply_rotary(q, freqs_cis)
            k = self.rotary.apply_rotary(k, freqs_cis)

        else:
            pos_q = torch.linspace(0, 1, ctx_q, device=q.device)
            pos_k = torch.linspace(0, 1, ctx_k, device=k.device)
            freqs_cis_q = self.rotary(pos_q)
            freqs_cis_k = self.rotary(pos_k)
            q = self.rotary.apply_rotary(q, freqs_cis_q)
            k = self.rotary.apply_rotary(k, freqs_cis_k)

        if Multihead.blend:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            w = F.dropout(w, p=self.dropout, training=self.training)
            out = torch.matmul(w, v)
            cos_w, cos_qk = self.compute_cosine_attention(q, k, v, mask)
            blend = torch.sigmoid(self.factor)
            out = blend * cos_w + (1 - blend) * out
            qk = blend * cos_qk + (1 - blend) * qk

        if Multihead.cos:
            out, qk = self.compute_cosine_attention(q, k, v, mask)

        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if self.use_betweenness:
                batch, heads, seq_len, head_dim = q.shape
                q_reshaped = q.reshape(batch * heads, seq_len, head_dim)
                betweenness = self.betweenness.compute_betweenness(q_reshaped)
                betweenness = betweenness.view(batch, heads, seq_len)
                betw_bias = betweenness.unsqueeze(-1)
                qk = qk + betw_bias

            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            w = F.dropout(w, p=self.dropout, training=self.training)
            out = torch.matmul(w, v)

        out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach() if self.training else qk
        return out, qk

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, cross_attention: bool = False, act = "relu"):
        super().__init__()
        self.dims = dims
        self.head = head
        self.cross_attention = cross_attention
        self.dropout = 0.1

        self.blend_xa = nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        self.blend = torch.sigmoid(self.blend_xa)

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = Multihead(dims=dims, head=head)
        self.attnb = Multihead(dims=dims, head=head) if cross_attention else None
    
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(normalized_shape=dims)    
        self.lnb = RMSNorm(normalized_shape=dims) if cross_attention else None
        self.lnc = RMSNorm(normalized_shape=dims) 

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        mask = mask if isinstance(self, TextDecoder) else None
        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnb and xa is not None:
            cross_out = self.attnb(self.lnb(x), xa, kv_cache=kv_cache)[0]
            x = self.blend * x + (1 - self.blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class AudioEncoder(nn.Module):
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act: str = "relu"):
        super().__init__()

        self._counter = 0
        self.use_betweenness = False  
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.mels = mels
        self.ctx = ctx
        self.dropout = 0.1
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(),
                   "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.blend_sw = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.blend = torch.sigmoid(self.blend_sw)
        self.ln_enc = RMSNorm(normalized_shape=dims)
        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        if self.use_betweenness:
            self.betweenness = BetweennessModule(dim=dims, window_size=1, adjustment_scale=0.5)

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1)
        )
        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(ctx),
        )

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False, act="relu")
                    for _ in range(layer)]) if layer > 0 else None)

    def forward(self, x, w) -> Tensor:
        if x is not None:
            if w is not None:
                x_spec = self.se(x).permute(0, 2, 1)
                w_wave = self.we(w).permute(0, 2, 1)
                if self._counter < 1:
                    plot_waveform_and_spectrogram(w, x)

                x = (x_spec + self.positional_embedding).to(x.dtype)
                w = w_wave
                x = self.blend * x + (1 - self.blend) * w
            else:
                x = self.se(x)
                x = x.permute(0, 2, 1)
                assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                x = (x + self.positional_embedding).to(x.dtype)
        else:
            assert w is not None, "You have to provide either x or w"
            x = self.we(w).permute(0, 2, 1)
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.dtype)

        if self.use_betweenness:
            be = self.betweenness.compute_betweenness(x)
            x = x + be.unsqueeze(-1) 

        for block in chain(self.blockA or []):
            x = block(x)
        self._counter += 1
        return self.ln_enc(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer):
        super().__init__()
        
        head_dim = dims // head
        self.ctx = ctx
        self.dropout = 0.1

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims))
        self.ln_dec = RMSNorm(normalized_shape=dims)
        self.rotary = Rotary(dim=head_dim, learned_freq=True)
        
        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False) for _ in range(layer)]) if layer > 0 else None)

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, kv_cache=None) -> Tensor:
        
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        ctx = x.shape[1]
        freqs_cis = self.rotary(ctx)
        x = self.rotary.apply_rotary(x, freqs_cis)

        x = x.to(xa.dtype)
        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=self.mask, kv_cache=kv_cache)
            
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param  

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.encoder_idx,
            act=param.act,
        )

        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.decoder_idx,
        )

        all_head = torch.zeros(self.param.decoder_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.decoder_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.decoder_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(input_ids, audio_features)

    @torch.autocast(device_type="cuda")
    def forward(self, 
        input_features: torch.Tensor=None, 
        waveform: Optional[torch.Tensor]=None,
        input_ids=None, 
        labels=None, 
        decoder_inputs_embeds=None,
        ) -> Dict[str, torch.Tensor]:
        
        if input_ids is None and decoder_inputs_embeds is None:
            if labels is not None:
                input_ids = shift_tokens_right(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id)
            else:
                raise ValueError("You have to provide either decoder_input_ids or labels")
            
        if input_features is not None:    
            if waveform is not None:
                encoded_audio = self.encoder(x=input_features, w=waveform)
            else:
                encoded_audio = self.encoder(x=input_features, w=None)
        elif waveform is not None:
            encoded_audio = self.encoder(x=None, w=waveform)
        else:
            raise ValueError("You have to provide either input_features or waveform")


        logits = self.decoder(input_ids, encoded_audio)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "labels": labels, "input_ids": input_ids, "audio_features": encoded_audio}

    @property
    def device(self):
        return next(self.parameters()).device

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.param.text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def save_adaptive_output(module, _, output):
            if isinstance(output, tuple) and len(output) == 2:
                tensor_output, cache_updates = output
                module_k = f"{module}_k"
                module_v = f"{module}_v"
                if module_k not in cache or tensor_output.shape[1] > self.param.text_ctx:
                    cache[module_k] = cache_updates["k_cache"]
                    cache[module_v] = cache_updates["v_cache"]
                else:
                    cache[module_k] = torch.cat([cache[module_k], cache_updates["k_cache"]], dim=1).detach()
                    cache[module_v] = torch.cat([cache[module_v], cache_updates["v_cache"]], dim=1).detach()
                return tensor_output
            return output
        
        def install_hooks(layer: nn.Module):
            if isinstance(layer, Multihead):
                hooks.append(layer.k.register_forward_hook(save_to_cache))
                hooks.append(layer.v.register_forward_hook(save_to_cache))
            self.encoder.apply(install_hooks)
        self.decoder.apply(install_hooks)
        return cache, hooks
    
    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, "Residual": 0,
                            "Multihead": 0, "MultiheadA": 0, "MultiheadB": 0, "MultiheadC": 0}

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
            elif isinstance(module, Multihead):
                nn.init.xavier_uniform_(module.q.weight)
                nn.init.zeros_(module.q.bias)
                nn.init.xavier_uniform_(module.k.weight)
                nn.init.xavier_uniform_(module.v.weight)
                nn.init.xavier_uniform_(module.o.weight)
                if module.o.bias is not None:
                    nn.init.zeros_(module.o.bias)
                self.init_counts["Multihead"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, SEBlock):
                nn.init.ones_(module.fc[0].weight)
                nn.init.zeros_(module.fc[0].bias)
                nn.init.ones_(module.fc[2].weight)
                nn.init.zeros_(module.fc[2].bias)
                self.init_counts["SEBlock"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                nn.init.xavier_uniform_(module.se[0].weight)
                nn.init.zeros_(module.se[0].bias)
                nn.init.xavier_uniform_(module.se[2].weight)
                nn.init.zeros_(module.se[2].bias)
                nn.init.xavier_uniform_(module.se[4].weight)
                nn.init.zeros_(module.se[4].bias)
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                nn.init.xavier_uniform_(module.attna.q.weight)
                nn.init.zeros_(module.attna.q.bias)
                nn.init.xavier_uniform_(module.attna.k.weight)
                nn.init.xavier_uniform_(module.attna.v.weight)
                nn.init.xavier_uniform_(module.attna.o.weight)
                if module.attna.o.bias is not None:
                    nn.init.zeros_(module.attna.o.bias)
                self.init_counts["Residual"] += 1    
                                                 
    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")

metric = evaluate.load(path="wer")

@dataclass
class DataCollator:
    extractor: Any
    tokenizer: Any
    decoder_start_token_id: Any

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        batch = {}

        if "input_features" in features[0]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch["input_features"] = self.extractor.pad(input_features, return_tensors="pt")["input_features"]
        if "waveform" in features[0]:
            waveforms = [f["waveform"] for f in features]
            fixed_len = 1500 * 160
            padded_waveforms = []
            for w in waveforms:
                if w.shape[-1] < fixed_len:
                    w = F.pad(w, (0, fixed_len - w.shape[-1]))
                else:
                    w = w[..., :fixed_len]
                padded_waveforms.append(w)
            batch["waveform"] = torch.stack(padded_waveforms)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def prepare_dataset(batch, input_features=True, waveform=True):
    audio = batch["audio"]
    fixed_len = 1500 * 160
    wav = torch.tensor(audio["array"]).float()
    if wav.shape[-1] < fixed_len:
        wav = F.pad(wav, (0, fixed_len - wav.shape[-1]))
    else:
        wav = wav[..., :fixed_len]
    if waveform:
        batch["waveform"] = wav.unsqueeze(0)
    if input_features:
        batch["input_features"] = extractor(wav.numpy(), sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

def compute_metrics(eval_pred):
    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if len(pred_ids) > 0:
        print("\nSample Predictions:")
        for idx in range(min(1, len(pred_ids))):
            print(f"  Example {idx+1}:")
            print(f"• Reference: {label_str[idx]}")
            print(f"• Prediction: {pred_str[idx]}")
        print("="*80 + "\n")

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    pred_flat = pred_ids.flatten()
    labels_flat = label_ids.flatten()
    mask = labels_flat != tokenizer.pad_token_id
    acc = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
    pre = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    rec = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)

    return {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        }

class MaxFactor(torch.optim.Optimizer):
    __version__ = "1.0"
    
    def __init__(self, params, lr=0.025, beta2_decay=-0.8, eps=(1e-10, 1e-4), d=1.0, 
                 weight_decay=0.025, gamma=0.99, max=False, min_lr=1e-7):
        
        print(f"Using MaxFactor optimizer v{self.__version__}")
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, max=max, min_lr=min_lr)
        super().__init__(params=params, defaults=defaults)

    def get_lr(self):
        """Return last-used learning rates for all parameter groups."""
        param_specific_lrs = []
        for group in self.param_groups:
            group_lrs = []
            for p in group["params"]:
                state = self.state[p]
                if "last_alpha" in state:
                    group_lrs.append(state["last_alpha"])
            if group_lrs:
                param_specific_lrs.append(sum(group_lrs) / len(group_lrs))
            else:
                param_specific_lrs.append(group["lr"])
        return param_specific_lrs
    
    def get_last_lr(self):
        return self.get_lr()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = p.new_zeros(row_shape)
                        state["col_var"] = p.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state = self.state[param]

                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[i]
                row_var, col_var, vi = row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, max(0.001, step_float ** group["beta2_decay"]))
                
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                alpha = max(eps2, (param.norm() / (param.numel() ** 0.5 + 1e-12)).item()) * rho_t

                state["last_alpha"] = alpha

                if group["weight_decay"] > 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi

                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                
                inf_norm = torch.norm(update, float('inf'))
                if inf_norm > 0:
                    update.div_(inf_norm.clamp_(min=eps1))
                
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                
                if param.dim() > 1:
                    max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                    param.add_(-alpha / denom * update.sign() * max_vals)
                else:
                    param.add_(-alpha / denom * update)
                
                state["step"] = step_t
                
        return loss

if __name__ == "__main__":

    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        encoder_idx=4,
        audio_dims=512,
        vocab=51865,
        text_ctx=512,
        text_head=4,
        decoder_idx=4,
        text_dims=512,
        decoder_start_token_id = 50258,
        pad_token_id = 50257,
        eos_token_id = 50257,   
        act = "gelu",
        )

    model = Echo(param).to('cuda')

    token=""
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", token=token, feature_size=128, sampling_rate=16000, do_normalize=True, return_tensors="pt", chunk_length=15)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe", token=token)
    data_collator = DataCollator(extractor=extractor,
        tokenizer=tokenizer, decoder_start_token_id=50258)

    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)

    dataset = DatasetDict()
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    dataset = dataset.map(function=prepare_dataset,
        remove_columns=list(next(iter(dataset.values())).features)).with_format(type="torch")

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=10000,
        save_steps=10000,
        eval_steps=1000,
        warmup_steps=1000,
        num_train_epochs=1,
        logging_steps=100,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        eval_on_start=False,
        # optim="adafactor",
        save_safetensors=True,
    )

    optimizer = MaxFactorA(model.parameters(), lr = 0.025,
    beta2_decay = -0.8,
    eps = (1e-10, 0.0001),
    d = 1,
    weight_decay = 0.025,
    gamma = 0.99,
    max = False,
    min_lr = 1e-7)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"].shuffle(seed=42).take(1000),
        eval_dataset=dataset["test"].take(100),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=extractor,
        optimizers=(optimizer, scheduler),  
    )

    model.init_weights()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    trainer.train(resume_from_checkpoint=False)



## pytorch loop

# def train(
#     model,
#     dataset,
#     data_collator,
#     tokenizer,
#     optimizer=None,
#     scheduler=None,
#     train_set=None,
#     eval_set=None,
#     epochs=3,
#     batch_size=1,
#     lr=2e-4,
#     device="cuda",
#     grad_accum_steps=1,
#     max_grad_norm=1.0,
#     log_dir="./output/logs",
#     save_best=True,
#     early_stopping_patience=None,
#     max_steps=10000,
#     eval_steps=1000,
# ):
#     from torch.utils.tensorboard import SummaryWriter
#     import os

#     writer = SummaryWriter(log_dir=log_dir)
#     model = model.to(device)
#     optimizer = optimizer
#     scheduler = scheduler
#     scaler = torch.amp.GradScaler('cuda')
#     train_set = dataset["train"]
#     eval_set = dataset["test"]

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
#     eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

#     best_wer = float("inf")
#     best_step = 0
#     patience_counter = 0
#     global_step = 0
#     running_loss = 0

#     train_iter = iter(train_loader)
#     pbar = tqdm(total=max_steps, desc="Training", dynamic_ncols=True)
#     model.train()
#     optimizer.zero_grad()


#     while global_step < max_steps:
#         try:
#             batch = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             batch = next(train_iter)
#         for k in batch:
#             if isinstance(batch[k], torch.Tensor):
#                 batch[k] = batch[k].to(device)
#         with torch.cuda.amp.autocast():
#             outputs = model(
#                 input_features=batch.get("input_features", None),
#                 waveform=batch.get("waveform", None),
#                 input_ids=None,
#                 labels=batch["labels"]
#             )
#             loss = outputs["loss"] / grad_accum_steps
#         scaler.scale(loss).backward()
#         running_loss += loss.item() * grad_accum_steps

#         if (global_step + 1) % grad_accum_steps == 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#             if scheduler is not None:
#                 scheduler.step()
#             writer.add_scalar("train/loss", loss.item() * grad_accum_steps, global_step)
#             writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)


#         pbar.set_postfix({
#             "loss": f"{loss.item() * grad_accum_steps:.4f}",
#             "lr": optimizer.param_groups[0]["lr"]
#         })
#         pbar.update(1)
#         global_step += 1

#         if global_step % eval_steps == 0 or global_step == max_steps:
#             model.eval()
#             all_preds, all_labels = [], []
#             eval_loss = 0
#             with torch.no_grad():
#                 for batch_eval in tqdm(eval_loader, desc=f"Eval@step{global_step}", leave=False):
#                     for k in batch_eval:
#                         if isinstance(batch_eval[k], torch.Tensor):
#                             batch_eval[k] = batch_eval[k].to(device)
#                     outputs = model(
#                         input_features=batch_eval.get("input_features", None),
#                         waveform=batch_eval.get("waveform", None),
#                         input_ids=None,
#                         labels=batch_eval["labels"]
#                     )
#                     logits = outputs["logits"]
#                     labels = batch_eval["labels"]
#                     loss = outputs["loss"]
#                     eval_loss += loss.item()
#                     preds = torch.argmax(logits, dim=-1)
#                     labels_for_decode = labels.clone()
#                     labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
#                     all_preds.extend(preds.cpu().numpy())
#                     all_labels.extend(labels_for_decode.cpu().numpy())
#             avg_eval_loss = eval_loss / len(eval_loader)
#             pred_str = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
#             label_str = tokenizer.batch_decode(all_labels, skip_special_tokens=True)

#             if len(all_preds) > 0:
#                 print("\nSample Predictions:")
#                 for idx in range(min(1, len(all_preds))):
#                     print(f"  Example {idx+1}:")
#                     print(f"• Reference: {label_str[idx]}")
#                     print(f"• Prediction: {pred_str[idx]}")
#                 print("="*80 + "\n")

#             wer = 100 * metric.compute(predictions=pred_str, references=label_str)
#             writer.add_scalar("eval/loss", avg_eval_loss, global_step)
#             writer.add_scalar("eval/wer", wer, global_step)
#             # scheduler.step(avg_eval_loss)
#             scheduler.step()
#             lr = scheduler.get_last_lr()[0]
#             pbar.set_postfix({
#                 "loss": f"{loss.item() * grad_accum_steps:.4f}",
#                 "lr": lr,
#                 "eval_wer": f"{wer:.2f}"
#             })
#             print(f"\nStep {global_step}: eval loss {avg_eval_loss:.4f}, WER {wer:.2f}")

#             # Save best model
#             if save_best and wer < best_wer:
#                 best_wer = wer
#                 best_step = global_step
#                 torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
#                 print(f"Best model saved at step {global_step} with WER {wer:.2f}")

#             # Early stopping
#             if early_stopping_patience is not None:
#                 if wer < best_wer:
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= early_stopping_patience:
#                         print(f"Early stopping at step {global_step}")
#                         break
#             model.train()
#             lr = scheduler.get_last_lr()[0]
#             writer.add_scalar("train/lr", lr, global_step)
#             pbar.set_postfix({
#                 "loss": f"{loss.item() * grad_accum_steps:.4f}",
#                 "lr": lr,
#                 "eval_wer": f"{wer:.2f}"
#             })

#     print(f"Training complete. Best WER: {best_wer:.2f} at step {best_step}")
#     writer.close()

# if __name__ == "__main__":

#     param = Dimensions(
#         mels=128,
#         audio_ctx=1500,
#         audio_head=4,
#         encoder_idx=4,
#         audio_dims=512,
#         vocab=51865,
#         text_ctx=512,
#         text_head=4,
#         decoder_idx=4,
#         text_dims=512,
#         decoder_start_token_id = 50258,
#         pad_token_id = 50257,
#         eos_token_id = 50257,   
#         act = "gelu",
#         )

#     model = Echo(param).to('cuda')

#     token=""
#     extractor = WhisperFeatureExtractor.from_pretrained(
#         "openai/whisper-small", token=token, feature_size=128, sampling_rate=16000, do_normalize=True, return_tensors="pt", chunk_length=15)
#     tokenizer = WhisperTokenizerFast.from_pretrained(
#         "openai/whisper-small", language="en", task="transcribe", token=token)
#     data_collator = DataCollator(extractor=extractor,
#         tokenizer=tokenizer, decoder_start_token_id=50258)

#     log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
#     os.makedirs(name=log_dir, exist_ok=True)

#     dataset = DatasetDict()
#     dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
#     dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
#     dataset = dataset.map(function=prepare_dataset,
#         remove_columns=list(next(iter(dataset.values())).features)).with_format(type="torch")
    
#     optimizer = MaxFactorA(model.parameters(), lr = 0.025,
#     beta2_decay = -0.8,
#     eps = (1e-10, 0.0001),
#     d = 1,
#     weight_decay = 0.025,
#     gamma = 0.99,
#     max = False,
#     min_lr = 1e-7)

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

#     train_set = dataset["train"],
#     eval_set = dataset["test"],

#     train(model=model, dataset=dataset, data_collator=data_collator, tokenizer=tokenizer, 
#     batch_size=1,
#     lr=2e-4,
#     device="cuda",
#     grad_accum_steps=1,
#     max_grad_norm=1.0,
#     log_dir="./output/logs",
#     save_best=True,
#     early_stopping_patience=None,
#     max_steps=10000,
#     eval_steps=1000,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     train_set=train_set,
#     eval_set=eval_set,
#     )

    # tensorboard --logdir ./output/logs


