import torch
import os
import pyworld as pw
import numpy as np
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, List, Dict
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from typing import Optional, Union, Tuple
from torch.nn.functional import scaled_dot_product_attention
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

def sinusoids(ctx, dims, max_tscale=10000):
    assert dims % 2 == 0
    pos = torch.log(torch.tensor(float(max_tscale))) / (dims // 2 - 1)
    tscales = torch.exp(-pos * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    position = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) 
    positional_embedding = nn.Parameter(position, requires_grad=True)
    return positional_embedding

class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = False, bias = True, batch_first = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bias, batch_first)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y

def cos_sim(q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
    q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
    k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
    qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
    qk_cosine = qk_cosine + mask
    weights = F.softmax(qk_cosine, dim=-1)
    out = torch.matmul(weights, v)
    return out

def scaled_relu(x, sequence_length):
    relu_output = torch.relu(x)
    return relu_output / sequence_length

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        tapprox += x**i / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

def taylor_masked(x, mask, order=2):
    tapprox = torch.zeros_like(x)
    unmasked = x.masked_select(mask) 
    approx_values = 1.0 + unmasked
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        approx_values += unmasked**i / factorial_i
    tapprox.masked_scatter_(mask, approx_values)
    sum_approx = torch.sum(tapprox, dim=-1, keepdim=True)
    toutput = tapprox / (sum_approx + 1e-9) 
    toutput = toutput * mask 
    return toutput

def taylor_softmax2(x, mask=None, order=2):
    if mask is None:
        tapprox = 1.0 + x
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            tapprox += x**i / factorial_i
        return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

    else:
        tapprox = torch.zeros_like(x)
        unmasked = x.masked_select(mask)
        tapprox = 1.0 + unmasked
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            tapprox += unmasked**i / factorial_i

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

def create_attention_mask(batch_size, ctx, is_causal=True, padding_mask=None, device=None):
    if is_causal:
        mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=0)
        mask = mask.expand(batch_size, 1, ctx, ctx)
    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device)
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
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head_dim = dims // head
        self.dims = dims
        self.q_module = nn.Linear(self.head_dim, self.head_dim)
        self.k_module = nn.Linear(self.head_dim, self.head_dim)
        self.v_module = nn.Linear(self.head_dim, self.head_dim)
        self.o_proj = nn.Linear(self.head_dim, self.head_dim)

    def _reshape_to_output(self, attn_output: Tensor) -> Tensor:
        batch, _, ctx, _ = attn_output.shape
        return attn_output.transpose(1, 2).contiguous().view(batch, ctx, self.dims)      

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
    seq_len_q = q.shape[1]
    seq_len_kv = k.shape[1]
    head_dim = dims // head

    q = q.view(batch_size, seq_len_q, head, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len_kv, head, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len_kv, head, head_dim).transpose(1, 2)
    return q, k, v

def create_qkv(dims, head, q, k, v, x, xa):
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
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

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

def track_x(new_x, operation=""): 
    """ track_x(x, "x") """
    x_id = [id(new_x)]
    if new_x is None:
        return new_x
    current_id = id(new_x)
    if current_id != x_id[0]:
        print(f"x FLOW: {x_id[0]} → {current_id} in {operation}")
        x_id[0] = current_id
    else:
        print(f"x REUSE: {current_id} in {operation}")
    return new_x

def track_xa(new_xa, operation=""): 
    """ track_xa(xa, "xa - decoder") """
    xa_id = [id(new_xa)] if new_xa is not None else [None]
    if new_xa is None:
        return new_xa
    current_id = id(new_xa)
    if current_id != xa_id[0]:
        print(f"xa FLOW: {xa_id[0]} → {current_id} in {operation}")
        xa_id[0] = current_id
    else:
        print(f"xa REUSE: {current_id} in {operation}")
    return new_xa

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

class feature_encoder(nn.Module):
    def __init__(self, mels, input_dims, dims, head, layer, act, features, feature=None, use_rope=False, spec_shape=None, debug=[], attend_feature=False, target_length=None):
        """
        Feature encoder for audio processing.
        """
        super().__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.attend_feature = attend_feature
        self.target_length = target_length
        self.feature = feature

        self.debug = debug
        act_fn = get_activation(act)

        if self.attend_feature:
            self.mlp = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))
        else:
            self.q, self.k, self.v, self.o, self.scale = None, None, None, None, None
            self.mlp = None

        self.spectrogram = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3), act_fn,
            Conv1d(dims, dims, kernel_size=3), act_fn,
            Conv1d(dims, dims, kernel_size=3, groups=dims), act_fn)

        self.waveform = nn.Sequential(
            Conv1d(1, dims//4, kernel_size=15, stride=4, padding=7), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=7, stride=2, padding=3), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=2, padding=2), act_fn)

        self.pitch = nn.Sequential(
            Conv1d(1, dims, kernel_size=7, stride=1, padding=3), act_fn,
            Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
            self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def rope(self, x, xa=None, mask=None, feats=None, feature=None, layer=None):
        if isinstance(x, int):
            ctx = x 
        elif isinstance(x, torch.Tensor):
            ctx = x.shape[1] if x.dim() > 1 else x.shape[0]
            batch, ctx, dims = x.shape[0], ctx, x.shape[-1]

            x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x

    def mel_scalar(self, freq: float) -> float:
        return 1127.0 * math.log(1.0 + freq / 700.0)

    def forward(self, x, xa=None, mask=None, feats=None, feature=None, layer=None, max_tscale=36000):
        target_length = x.shape[1] if self.target_length is None else self.target_length

        if feature == "pitch":
            xp = x.clone()
            enc_dict = feats if feats is not None else {}
            enc_dict = dict(enc_dict)  
            enc_dict["f0"] = xp
            feats = enc_dict
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.pitch(x).permute(0, 2, 1)
  
        if feature == "phase":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.pitch(x).permute(0, 2, 1)

        if feature == "waveform":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.waveform(x).permute(0, 2, 1)
            if target_length and x.shape[1] != self.target_length:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)
        
        if feature == "harmonics":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)

        if feature == "aperiodic":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)            

        if feature == "spectrogram":
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.spectrogram(x).permute(0, 2, 1)

        if self.use_rope:
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
            x = self.rope(x=x, xa=None, mask=None, feats=feats, feature=feature, layer=layer)
        else:
            max_tscale = x.shape[1] * 1000 if max_tscale is None else max_tscale
            x = x + self.positional(x.shape[1], x.shape[-1], max_tscale).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)

        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class OneShot(nn.Module):
    def __init__(self, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "aperiodic", "harmonics"]
        self.head = head
        self.head_dim = dims // head
        self.scale = 1.0 // len(features) if features else scale

        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims)

    def forward(self, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = self.q(x).view(B, L, self.head, self.head_dim).transpose(1,2)
        k = self.k(xa).view(B, K, self.head, self.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.head_dim)
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

def valid(default_value, *items):
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

def sinusoids(ctx, dims, max_tscale=10000):
    assert dims % 2 == 0
    pos = torch.log(torch.tensor(float(max_tscale))) / (dims // 2 - 1)
    tscales = torch.exp(-pos * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    position = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) 
    positional_embedding = nn.Parameter(position, requires_grad=True)
    return positional_embedding

class SelfCriticalRL(nn.Module):
    def __init__(self, model, tokenizer, reward_fn):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

    def forward(self, input_ids, features, labels=None, max_len=128, feature_name="spectrogram"):

        with torch.no_grad():
            greedy_ids = self.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len)
        greedy_text = [self.tokenizer.decode(ids) for ids in greedy_ids]
        sampled_ids = self.model.generate(input_ids=input_ids, **{feature_name: features}, max_length=max_len, do_sample=True, top_k=5)
        sampled_text = [self.tokenizer.decode(ids) for ids in sampled_ids]
        
        rewards = []
        baseline = []
        for s, g, ref in zip(sampled_text, greedy_text, labels):
            ref_text = self.tokenizer.decode(ref)
            rewards.append(self.reward_fn(s, ref_text))
            baseline.append(self.reward_fn(g, ref_text))
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        baseline = torch.tensor(baseline, device=device, dtype=torch.float)
        advantage = rewards - baseline
        logits = self.model(input_ids=sampled_ids, **{feature_name: features})["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_seq = torch.gather(log_probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_sum = log_probs_seq.sum(dim=1)
        loss = -(advantage * log_probs_sum).mean()
        return loss

class SelfTrainingModule(nn.Module):
    def __init__(self, model, tokenizer, quality_fn=None, threshold=0.8):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.quality_fn = quality_fn
        self.threshold = threshold

    def generate_pseudo_labels(self, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        with torch.no_grad():
            pred_ids = self.model.generate(input_ids=unlabeled_batch, **{feature_name: features}, max_length=max_len)

        if self.quality_fn is not None:
            quality_scores = self.quality_fn(pred_ids, self.model, features)
            mask = quality_scores > self.threshold
            pred_ids = pred_ids[mask]
        return pred_ids

    def forward(self, unlabeled_batch, features, max_len=128, feature_name="spectrogram"):
        pseudo_labels = self.generate_pseudo_labels(unlabeled_batch, features, max_len, feature_name=feature_name)
        logits = self.model(input_ids=unlabeled_batch, **{feature_name: features}, labels=pseudo_labels)["logits"]
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
    for i in range(len(hyp_words)+1):
        d[i][0] = i
    for j in range(len(ref_words)+1):
        d[0][j] = j
    for i in range(1, len(hyp_words)+1):
        for j in range(1, len(ref_words)+1):
            if hyp_words[i-1] == ref_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
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

def tokenize_pitch(pitch_features, target_length):
    pitch_len = pitch_features.shape[-1]
    token_len = target_length
    if pitch_len > token_len:
        pitch_tokens = F.adaptive_avg_pool1d(pitch_features, token_len)
    else:
        pitch_tokens = F.interpolate(pitch_features, token_len)
    return pitch_tokens

def load_wave(wave_data, sample_rate=16000):

    if isinstance(wave_data, str):
        waveform, sample_rate = torchaudio.load(uri=wave_data, normalize=False)
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

def extract_features(batch, tokenizer, waveform=False, spec=False, pitch_tokens=False, pitch=False, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, phase_mod=False, crepe=False, aperiodics=False, dummy=False):

    torch_windows = {
        'hann': torch.hann_window,
        'hamming': torch.hamming_window,
        'blackman': torch.blackman_window,
        'bartlett': torch.bartlett_window,
        'ones': torch.ones,
        None: torch.ones,
    }
    if dummy:
        return {
            "spectrogram": torch.zeros((1, 128, 100)),
            "f0": torch.zeros((1, 100)),
            "pitch_tokens": torch.zeros((1, 100)),
            "pitch": torch.zeros((1, 100)),
            "harmonics": torch.zeros((1, 128, 100)),
            "aperiodics": torch.zeros((1, 128, 100)),
            "crepe_time": None,
            "crepe_frequency": None,
            "crepe_confidence": None,
            "crepe_activation": None,
        }

    audio = batch["audio"]
    sample_rate = audio["sampling_rate"]
    labels = tokenizer.encode(batch["transcription"])
    wav = load_wave(wave_data=audio, sample_rate=sample_rate)
    
    def crepe_predict(wav, sample_rate, viterbi=False):
        import torchcrepe
        wav = wav.numpy().astype(np.float32)
        time, frequency, confidence, activation = torchcrepe.predict(
            wav, sample_rate=sample_rate, viterbi=viterbi)
        crepe_time = torch.from_numpy(time)
        crepe_frequency = torch.from_numpy(frequency)
        crepe_confidence = torch.from_numpy(confidence)
        crepe_activation = torch.from_numpy(activation)
        return crepe_time, crepe_frequency, crepe_confidence, crepe_activation

    if crepe:
        crepe_time, crepe_frequency, crepe_confidence, crepe_activation = crepe_predict(wav, sample_rate, viterbi=True)

    else:
        crepe_time = None
        crepe_frequency = None
        crepe_confidence = None
        crepe_activation = None

    def spectrogram(wav, sample_rate, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
        if isinstance(window_fn, str):
            window_fn = torch_windows[window_fn]
        if window_fn is None:
            window_fn = torch.ones(n_fft)
        if isinstance(window_fn, torch.Tensor):
            window_fn = window_fn.to(device)
        return torchaudio.functional.spectrogram(
            wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            window=window_fn, center=True, pad_mode="reflect", power=1.0)

    def mel_spectrogram(wav, sample_rate):

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
        transform = torchaudio.transforms.MelSpectrogram(**spectrogram_config)
        mel_spectrogram = transform(wav)
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spectrogram_tensor = (log_mel + 4.0) / 4.0
        return spectrogram_tensor

    if spec: 
        spectrogram_tensor = mel_spectrogram(wav, sample_rate)

    def mfcc(wav, sample_rate, n_mels=128, n_fft=1024, hop_length=256, window_fn=torch.hann_window):
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
        mfcc_tensor = transform(wav)
        return mfcc_tensor

    def harmonics_and_aperiodics(wav, f0, t, sample_rate):
        import pyworld as pw
        wav_np = wav.numpy().astype(np.float64)
        sp = pw.cheaptrick(wav_np, f0, t, sample_rate, fft_size=256)
        ap = pw.d4c(wav_np, f0, t, sample_rate, fft_size=256)
        harmonic_tensor = torch.from_numpy(sp)
        aperiodic_tensor = torch.from_numpy(ap)
        harmonic_tensor = harmonic_tensor[:, :128].contiguous().T
        aperiodic_tensor = aperiodic_tensor[:, :128].contiguous().T
        harmonic_tensor = torch.where(harmonic_tensor == 0.0, torch.zeros_like(harmonic_tensor), harmonic_tensor / 1.0)
        aperiodic_tensor = torch.where(aperiodic_tensor == 0.0, torch.zeros_like(aperiodic_tensor), aperiodic_tensor / 1.0)
        return harmonic_tensor, aperiodic_tensor

    if pitch or pitch_tokens or harmonics or aperiodics:
        wavnp = wav.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

    if pitch_tokens:
        wav = torch.from_numpy(wavnp)
        t2 = torch.from_numpy(t)
        audio_duration = len(wav) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t2, token_starts, side="left")
        end_idx = torch.searchsorted(t2, token_ends, side="right")
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
        p_tensor = torchaudio.functional.detect_pitch_frequency(wav, sample_rate).unsqueeze(0)
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
        wave_tensor = wav
    else:
        wave_tensor = None

    if dummy:   
        if spectrogram_tensor is not None:
            dummy_tensor = torch.ones_like(spectrogram_tensor)
        elif p_tensor is not None:
            dummy_tensor = torch.ones_like(p_tensor) 
        elif pitch_tokens_tensor is not None:
            dummy_tensor = torch.ones_like(pitch_tokens_tensor)
        else:
            batch_size = 128
            seq_len = 1024
            dummy_tensor = torch.ones(batch_size, seq_len)
            dummy_tensor = dummy_tensor.to(device)
    else:
        dummy_tensor = None
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
        print(f"['dummy']: {dummy_tensor.shape if dummy else None}")

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
        "dummy": dummy_tensor if dummy else None,
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

class tgate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(Linear(dims, num_types), nn.Softmax(dim=-1))
    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        cgate = torch.sum(gates * types.unsqueeze(2), dim=-1)
        return cgate

def get_feature_encoder(feature: str, mels: int, input_dims: int, dims: int, head: int, layer: int, act=None, features=None) -> nn.Module:
    if feature == "spectrogram":
        return FEncoder(mels=mels, input_dims=input_dims, dims=dims, head=head)
    elif feature == "waveform":
        return WEncoder(input_dims, dims, head, layer, act, feature, features)
    elif feature == "pitch":
        return PEncoder(input_dims, dims, head, layer, act, feature, features)
    else:
        raise ValueError(f"Unknown feature type: {feature}")

class FEncoder(nn.Module):
    def __init__(self, mels, input_dims, dims, head, layer, act, feature, features, use_rope=False, spec_shape=None, debug=[]):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        self.debug = debug
        self.feature = feature
        self.mels = mels
        self.input_dims = input_dims
        act_fn = get_activation(act)

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

    def apply_rope_to_features(self, x, xa=None, mask=None, feats=None, feature="audio", layer="FEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)

        return x

    def forward(self, x, xa=None, mask=None, feats=None, feature="audio", layer="FEncoder"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, xa=xa, mask=mask, feats=feats, feature=feature, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        print(f"feature encoder: {x.shape} {feature}") if "fencoder" in self.debug else None
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

    def apply_rope_to_features(self, x, xa=None, mask=None, feats=None, feature="waveform", layer="WEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, xa=None, mask=None, feats= None, feature="waveform", layer = "WEncoder"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.target_length and x.shape[1] != self.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.target_length).transpose(1, 2)
        if self.use_rope:
            x = self.apply_rope_to_features(x, xa=xa, mask=mask, feats=feats, feature=feature, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        print(f"waveform encoder: {x.shape} {feature}") if "fencoder" in self.debug else None
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, act, use_rope=False, debug=[], one_shot=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dims = dims
        self.dropout = 0.01
        self.use_rope = use_rope
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

        self.pitch_encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=7, stride=1, padding=3), act_fn,
            Conv1d(dims, dims, kernel_size=5, stride=1, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        if use_rope:
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)
        
    def rope_to_feature(self, x, xa=None, mask=None, feats=None, feature="pitch", layer="PEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)
        x = self.rope.apply_rotary(x, freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, xa=None, mask=None, feats= None, feature="pitch", layer="PEncoder"):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.pitch_encoder(x).permute(0, 2, 1)

        if self.use_rope:
            x = self.rope_to_feature(x, xa=xa, mask=mask, feats=feats, feature=feature, layer=layer)
    
        x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        if self.mlp is not None:
            x = self.mlp(x)

        if self.attend_pitch:
            if xa is not None:
                q, k, v = create_qkv(self.q, self.k, self.v, x=xa, xa=x, head=self.head)
                out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
                x = x + out

        x = self.norm(x)    
        print(f"Pitch encoder: {x.shape} {feature}") if "fencoder" in self.debug else None
        return x

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
                    padded.append(pad_item)
                batch[key] = torch.stack(padded)
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

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None):
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(i) for i in seq if i not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(i) for i in ids if i not in (-100, pad_token_id, bos_token_id, eos_token_id)]

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
        for i in range(min(num_samples, len(pred_ids))):

            print(f"Pred tokens: {pred_ids[i]}")
            print(f"Label tokens: {label_ids[i]}")
            print(f"Pred: '{pred_str[i]}'")
            print(f"Label: '{label_str[i]}'")
            print("-" * 40)
    
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

