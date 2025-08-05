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
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

def cos_sim(q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
    q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
    k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
    qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
    qk_cosine = qk_cosine + mask
    weights = F.softmax(qk_cosine, dim=-1)
    out = torch.matmul(weights, v)
    return out

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
    # mask[i, j] = 1 if j in [i-window+1, i], else 0
    idxs = torch.arange(q_len, device=device).unsqueeze(1)
    jdxs = torch.arange(k_len, device=device).unsqueeze(0)
    mask = (jdxs >= (idxs - window + 1)) & (jdxs <= idxs)
    return mask.float()  # shape: (q_len, k_len)

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

# def calculate_attention(q, k, v, mask=None, temp=1.0):
#     scaled_q = q
#     if temp != 1.0 and temp > 0:
#         scaled_q = q * (1.0 / temp)**.5
#     out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
#     return out

# class LocalOut(nn.Module):
#     def __init__(self, dims: int, head: int):
#         super().__init__()
#         head_dim = dims // head
#         self.head_dim = head_dim
#         self.query_module = nn.Linear(head_dim, head_dim)
#         self.key_module = nn.Linear(head_dim, head_dim)
#         self.value_module = nn.Linear(head_dim, head_dim)
#         self.out_proj = nn.Linear(head_dim, head_dim)
    
#     def _reshape_to_output(self, x):
#         return x

# class attentiona(nn.Module):
#     def __init__(self, dims: int, head: int, max_iter: int = 3, threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1, temp = 1.0):
#         super(attentiona, self).__init__()
#         self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
#         self.dims = dims
#         self.head = head
#         self.head_dim = dims // head
#         self.max_iter = max_iter
#         self.threshold = nn.Parameter(torch.tensor(threshold))
#         self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        
#         self.factor = nn.Parameter(torch.tensor(factor))
#         self.lnc = nn.LayerNorm(self.head_dim, bias=False)
#         self.lnd = nn.LayerNorm(self.head_dim, bias=False)     
#         self.attn_local = LocalOut(self.head_dim)   

#     def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
#         z = default(xa, x)
#         q, k, v = create_qkv(self.dims, self.head, self.q, self.k, self.v, self.lna(x), self.lna(z))    

#         iteration = 0
#         temp = self.temp.item()
#         prev_out = torch.zeros_like(q)
#         attn_out = torch.zeros_like(q)
#         threshold = self.threshold.item()
#         factor = self.factor.item()
#         qcur = q

#         while iteration < self.max_iter:
#             eff_span = min(qcur.shape[1], k.shape[1])
#             if xa is not None:
#                 eff_span = min(eff_span, xa.shape[1])
#             if eff_span == 0: 
#                 break

#             qiter = qcur[:, :, :eff_span, :]
#             kiter = k[:, :, :eff_span, :]
#             viter = v[:, :, :eff_span, :]
#             q = self.attn_local.query_module(qiter)
#             k = self.attn_local.key_module(kiter)
#             v = self.attn_local.value_module(viter)

#             iter_mask = None
#             if mask is not None:
#                 if mask.dim() == 4: 
#                     iter_mask = mask[:, :, :eff_span, :eff_span]
#                 elif mask.dim() == 2: 
#                     iter_mask = mask[:eff_span, :eff_span]

#             attn_iter = calculate_attention(
#                 self.lnc(q), self.lnd(k), v,
#                 mask=iter_mask, temp=temp)

#             iter_out = torch.zeros_like(qcur)
#             iter_out[:, :, :eff_span, :] = attn_iter
#             diff = torch.abs(iter_out - prev_out).mean()
#             dthresh = threshold + factor * diff
#             if diff < dthresh and iteration > 0:
#                 attn_out = iter_out
#                 break

#             prev_out = iter_out.clone()
#             qcur = qcur + iter_out
#             attn_out = iter_out
#             iteration += 1
#             temp += 0.005

#         output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
#         return self.o(output), None

    # def _slide_win_local(self, x: Tensor, win_size: int, span_len: int, mask: Optional[Tensor] = None) -> Tensor:

    #     batch, ctx, dims = x.shape
    #     output = torch.zeros_like(x)
    #     num_win = (ctx + win_size - 1) // win_size

    #     for i in range(num_win):
    #         qstart = i * win_size
    #         qend = min(qstart + win_size, ctx)
    #         win_qlen = qend - qstart
    #         if win_qlen == 0: 
    #             continue

    #         kstart = max(0, qend - span_len)
    #         kend = qend
    #         qwin = x[:, qstart:qend, :]
    #         kwin = x[:, kstart:kend, :]

    #         win_mask = None
    #         if mask is not None:
    #             if mask.dim() == 4:
    #                 win_mask = mask[:, :, qstart:qend, kstart:kend]
    #             elif mask.dim() == 2:
    #                 win_mask = mask[qstart:qend, kstart:kend]

    #         attn_out, _ = self._focus(x=qwin, xa=kwin, mask=win_mask)
    #         output[:, qstart:qend, :] = attn_out
    #     return output

    # def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
    #             use_sliding_win: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
    #     if use_sliding_win:
    #         return self._slide_win_local(x, win_size, span_len, mask)
    #     else:
    #         output, _ = self._focus(x, xa, mask)
    #         return output
   
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val  # pyright: ignore[reportIndexIssue]
        v_out[:, :, input_pos] = v_val # pyright: ignore[reportIndexIssue]

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
        xa_id[0] = current_id  # pyright: ignore[reportArgumentType, reportCallIssue]
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
    return GenerationConfig(    # type: ignore
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
            # self.q, self.k, self.v, self.o, self.scale = qkv_init(dims, head)
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
            # if spec_shape is not None:
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
            self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape) # type: ignore
        else:
            self.rope = None # type: ignore
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
        x = self.rope.apply_rotary(x, freqs)  # pyright: ignore[reportOptionalSubscript, reportAttributeAccessIssue]
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
            # xp = self.mel_scalar(xp.mean())
            # print(f"Using pitch scalar: {xp}")
            # max_tscale = xp*300
            # print(f"Using max_tscale: {max_tscale}")
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

        # if self.attend_feature:
        #     xa = feats[feature]  # pyright: ignore[reportOptionalSubscript]
        #     if xa is not None:
        #         q, k, v = create_qkv(self.q, self.k, self.v, x=xa, xa=x, head=self.head)
        #         out, _ = calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True)
        #         x = x + out

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
        for s, g, ref in zip(sampled_text, greedy_text, labels): # type: ignore
            ref_text = self.tokenizer.decode(ref)
            rewards.append(self.reward_fn(s, ref_text))
            baseline.append(self.reward_fn(g, ref_text))
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        baseline = torch.tensor(baseline, device=device, dtype=torch.float)
        advantage = rewards - baseline
        logits = self.model(input_ids=sampled_ids, **{feature_name: features})["logits"]  # logits: [batch, sampled_seq_len, vocab_size]
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
    return -wer  # negative WER as reward

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
        sample_rate = wave_data["sampling_rate"]  # noqa: F841
    else:
        raise TypeError("Invalid wave_data format.")
    return waveform

def world_to_mel(sp, ap, sample_rate=16000, n_mels=128):
    import librosa
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()
    sp_mel = torch.matmul(sp, mel_basis.T)  # (frames, 128)
    ap_mel = torch.matmul(ap, mel_basis.T)  # (frames, 128)
    return sp_mel, ap_mel

def extract_features(batch, tokenizer, waveform=False, spec=False, pitch_tokens=False, pitch=False, harmonics=False, sample_rate=16000, hop_length=256, mode="mean", debug=False, phase_mod=False, crepe=False, aperiodics=False, dummy=False):

    # import torch
    # import torchaudio
    # import torchaudio.functional as F
    # import torchaudio.transforms as T

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
    # audio_length = len(audio["array"]) / audio["sampling_rate"]
    labels = tokenizer.encode(batch["transcription"])
    # sentence_length = len(batch["transcription"]) 
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

    # def compute_pitch(wav, sample_rate, hop_length=256):
    #     # pitch = F.detect_pitch_frequency(wav, sample_rate)
    #     # f0 = pitch
    #     import pyworld as pw
    #     wav_np = wav.numpy().astype(np.float64)
    #     f0, t = pw.dio(wav_np, sample_rate, frame_period=hop_length / sample_rate * 1000)
    #     f0 = pw.stonemask(wav_np, f0, t, sample_rate)
    #     return f0, t

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
            lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i]) # type: ignore
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
        omega = 2 * torch.pi * f0_tensor # type: ignore
        dphi = omega * tframe
        phi = torch.cumsum(dphi, dim=0) + phi0
        phase = torch.remainder(phi, 2 * torch.pi)
    else:
        phase = None

    if pitch:
        p_tensor = torchaudio.functional.detect_pitch_frequency(wav, sample_rate)
        # p_tensor = torch.from_numpy(f0_np)
        # p_tensor = p_tensor.unsqueeze(0) 
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

# class PEncoder(nn.Module): # pitch encoder
#     def __init__(self, dims: int, head: int, layer: int, kernel_size: int, act: str, 
#                  max_seq_len: int, input_dims: int = 1, use_rope=False): 
#         super().__init__()
        
#         self.head = head
#         self.head_dim = dims // head
#         self.dims = dims
#         self.use_rope=use_rope
#         self.dropout_rate = 0.01 
#         act_fn = get_activation(act)

#         self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, dims)) 

#         if use_rope:
#                 self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)# type: ignore
#         else:
#             self.rope = None
#             self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)

#         self.attend_pitch = False 
#         if self.attend_pitch:
#             self.mlp = nn.Sequential(
#                 nn.Linear(dims, dims),
#                 nn.ReLU(),
#                 nn.Linear(dims, dims),
#             )
#         else:
#             self.mlp = None

#         self.pitch_encoder = nn.Sequential(
#             nn.Conv1d(input_dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2), act_fn,
#             nn.Conv1d(dims, dims, kernel_size=kernel_size - 2, stride=1, padding=(kernel_size - 2) // 2), act_fn,
#             nn.Conv1d(dims, dims, kernel_size=kernel_size - 4, stride=1, padding=(kernel_size - 4) // 2, groups=dims), act_fn
#         )
    
#         def rope_to_feature(self, x, xa=None, mask=None, feats=None, feature="pitch", layer="PEncoder"):
#             batch, ctx, dims = x.shape
#             x = x.view(batch, ctx, head, self.head_dim).permute(0, 2, 1, 3)
#             freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer) # type: ignore
#             x = self.rope.apply_rotary(x, freqs)# type: ignore
#             x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
#             return x

#         self.norm = nn.LayerNorm(dims) 

#     def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
#                 feats: Optional[Any] = None, feature: str = "pitch", layer: str = "PEncoder", 
#                 audio_duration: Optional[float] = None, sample_rate: Optional[int] = None, 
#                 labels_len: Optional[int] = None, f0_np: Optional[np.ndarray] = None, 
#                 t2_np: Optional[np.ndarray] = None, mode: str = "mean") -> Tensor:
        
#         if x.dim() == 2 and feature == "pitch":
#             x_processed = x.unsqueeze(1) # Input to pitch_encoder: (batch_size, 1, num_pitch_tokens)
#             x_processed = self.pitch_encoder(x_processed)             # Output: (batch_size, dims, num_pitch_tokens)
#             x = x_processed.permute(0, 2, 1) # Reassign to x for consistency

#         if self.use_rope:
#             pass # Placeholder for RoPE application
    
#         seq_len = x.shape[1]
#         # x = x + self.positional_encoding[:, :seq_len, :] 
#         x = x + sinusoids(x.shape[1], x.shape[-1], 36000).to(device, dtype)

#         if self.mlp is not None:
#             x = self.mlp(x)

#         x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.norm(x)    

#         return x

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

        # raw_train  = load_dataset("mozilla-foundation/common_voice_17_0", "en", token=token, split="train", trust_remote_code=True, streaming=True).rename_column("sentence", "transcription")
        # raw_test = load_dataset("mozilla-foundation/common_voice_17_0", "en", token=token, split="test", trust_remote_code=True, streaming=True).rename_column("sentence", "transcription").take(1000)

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
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape) # type: ignore
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, xa=None, mask=None, feats=None, feature="audio", layer="FEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)# type: ignore
        x = self.rope.apply_rotary(x, freqs)# type: ignore
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

class WEncoder(nn.Module): # waveform encoder
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
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)# type: ignore
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, xa=None, mask=None, feats=None, feature="waveform", layer="WEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer)# type: ignore
        x = self.rope.apply_rotary(x, freqs)# type: ignore
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, xa=None, mask=None, feats= None, feature="waveform", layer = "WEncoder"):
        x = self.encoder(x).permute(0, 2, 1)  # (batch, time, dims)
        if self.target_length and x.shape[1] != self.target_length:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.target_length).transpose(1, 2)
        if self.use_rope:
            x = self.apply_rope_to_features(x, xa=xa, mask=mask, feats=feats, feature=feature, layer=layer)
        else:
            x = x + self.positional(x.shape[1], x.shape[-1], 10000).to(device, dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        print(f"waveform encoder: {x.shape} {feature}") if "fencoder" in self.debug else None
        return self.norm(x)

class PEncoder(nn.Module): # pitch encoder
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
                self.rope = rotary(dims=dims, head=head, radii=False, debug=[], use_pbias=False, axial=False, spec_shape=spec_shape)# type: ignore
        else:
            self.rope = None
            self.positional = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.norm = RMSNorm(dims)
        
    def rope_to_feature(self, x, xa=None, mask=None, feats=None, feature="pitch", layer="PEncoder"):
        batch, ctx, dims = x.shape
        x = x.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        freqs = self.rope(ctx, feats=feats, feature=feature, layer=layer) # type: ignore
        x = self.rope.apply_rotary(x, freqs)# type: ignore
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, ctx, dims)
        return x
        
    def forward(self, x, xa=None, mask=None, feats= None, feature="pitch", layer="PEncoder"):
        # f0=x
        # freqs = self.rope(f0.shape[1], feats=feats, feature=feature, layer=layer)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # if feature == "pitch":
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

        # x = nn.functional.dropout(x, p=self.dropout, training=self.training)
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
                max_len = max(len(l) for l in labels_list)  # noqa: E741
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
                # if key == "spectrogram":
                #     batch["spectrogram"] = batch[key]
        return batch

# import tiktoken
# import torch
# from torch.utils.data import Dataset, DataLoader

# class tokenize(Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):

#         self.input_ids = []
#         self.labels = []

#         token_ids = tokenizer.encode(txt, allowed_special={"<eos>"})

#         for i in range(0, len(token_ids) - max_length, stride):

#             input_chunk = token_ids[i:i + max_length]
#             target_chunk = token_ids[i + 1: i + max_length + 1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.labels.append(torch.tensor(target_chunk))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.labels[idx]

# def create_dataloader_v1(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = tokenize(txt, tokenizer, max_length, stride)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
#     return dataloader

# def custom_collate_fn(batch, tokenizer_pad_token_id):
#     max_len_in_batch = max(len(seq) for seq in batch)

#     padded_input_ids = []
#     attention_masks = []
#     for seq in batch:
#         padded_seq = F.pad(seq, (0, max_len_in_batch - len(seq)), value=tokenizer_pad_token_id)
#         attention_mask = torch.ones(max_len_in_batch)
#         attention_mask[len(seq):] = 0 
#         padded_input_ids.append(padded_seq)
#         attention_masks.append(attention_mask)

#     input_ids_tensor = torch.stack(padded_input_ids)
#     attention_mask_tensor = torch.stack(attention_masks)
#     labels_tensor = input_ids_tensor.clone() 

#     return {
#         'input_ids': input_ids_tensor,
#         'attention_mask': attention_mask_tensor,
#         'labels': labels_tensor
#     }

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# vocab_size = 50257
# output_dim = 256
# context_length = 1024

# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# batch_size = 8
# max_length = 4
# dataloader = create_dataloader_v1(
#     raw_text,
#     batch_size=batch_size,
#     max_length=max_length,
#     stride=max_length
# )

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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from typing import Optional, Tuple
# import numpy as np
# from torch.nn.functional import scaled_dot_product_attention
# from torch.cuda.amp import autocast
# from torch.nn import LayerNorm, Linear
# import logging

# logging.basicConfig(level=logging.WARNING)
# log = logging.getLogger(__name__)

# class ProjectionModule(nn.Module):
#     """
#     Projects input embeddings into query, key, or value representations
#     for multi-head attention, handling scaling for Q/K.
#     """
#     def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
#         """
#         Args:
#             dims: Input and output dimension.
#             head: Number of attention heads.
#             proj_type: Type of projection ("query", "key", "value").
#             use_bias: Whether to use bias in the linear layer.
#         """
#         super().__init__()
#         assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
#         assert proj_type in ["query", "key", "value"], \
#                f"proj_type must be 'query', 'key', or 'value', got {proj_type}"

#         self.dims = dims
#         self.head = head
#         self.head_dim = dims // head
#         self.proj_type = proj_type
#         self.scale = self.head_dim ** -0.5 if proj_type != "value" else 1.0
#         self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
#         self.init_weights()

#     def init_weights(self):
#         """Initialize projection weights."""
#         nn.init.normal_(tensor=self.proj.weight, std=0.02)
#         if self.proj.bias is not None:
#             nn.init.zeros_(tensor=self.proj.bias)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Applies projection, scaling (for Q/K), and reshapes for multi-head attention.

#         Args:
#             x: Input tensor of shape (batch, seq_len, dims).

#         Returns:
#             Projected tensor of shape (batch, head, seq_len, head_dim).
#         """
#         batch, seq_len, _ = x.shape
#         proj = self.proj(x)

#         proj = proj.view(batch, seq_len, self.head, self.head_dim).permute(0, 2, 1, 3)

#         if self.proj_type in ["query", "key"]:
#             proj = proj * self.scale
#         return proj

# def calculate_attention(
#     q: Tensor,
#     k: Tensor,
#     v: Tensor,
#     mask: Optional[Tensor] = None,
#     temperature: float = 1.0,
#     use_sdpa: bool = True,
#     is_causal: bool = False,
#     dropout_p: float = 0.0
# ) -> Tuple[Tensor, Optional[Tensor]]:
#     """
#     Calculates scaled dot-product attention.

#     Uses torch.nn.functional.scaled_dot_product_attention if use_sdpa is True
#     and inputs are compatible, otherwise falls back to manual implementation.

#     Args:
#         q: Query tensor (Batch, Heads, SeqLen_Q, HeadDim). Already scaled if needed.
#         k: Key tensor (Batch, Heads, SeqLen_K, HeadDim). Already scaled if needed.
#         v: Value tensor (Batch, Heads, SeqLen_K, HeadDim).
#         mask: Attention mask. Can be boolean (True means ignore) or float (-inf means ignore).
#               Shape should be broadcastable to (Batch, Heads, SeqLen_Q, SeqLen_K).
#         temperature: Softmax temperature scaling. Applied *before* softmax.
#         use_sdpa: Flag to attempt using PyTorch's optimized SDPA implementation.
#         is_causal: If True, applies a causal mask (for decoder self-attention).
#                    Used only if mask is None and use_sdpa is True.
#         dropout_p: Dropout probability for attention weights.

#     Returns:
#         A tuple containing:
#         - attn_output: Attention output tensor (Batch, Heads, SeqLen_Q, HeadDim).
#         - attn_weights: Attention weights tensor (Batch, Heads, SeqLen_Q, SeqLen_K),
#                         or None if SDPA implementation doesn't return them or if fallback used.
#                         *Note: SDPA's default doesn't return weights, requires specific backend support.*
#                         *Manual path always returns weights.*
#     """
#     batch_size, num_heads, q_len, head_dim = q.shape
#     k_len = k.size(2)

#     temp_scale = 1.0 / temperature if temperature > 0 else 1.0

#     attn_output, attn_weights = None, None

#     if use_sdpa:
#         try:
#             if temperature != 1.0:
#                  raise NotImplementedError("SDPA does not directly support temperature scaling. Use manual path or scale Q.")

#             attn_output = scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask=mask,
#                 dropout_p=dropout_p,
#                 is_causal=is_causal and mask is None
#             )
#             attn_weights = None
#             return attn_output, attn_weights
#         except (RuntimeError, NotImplementedError) as e:
#             log.warning(f"SDPA failed or not used ({e}), falling back to manual attention.")
#     attn_scores = torch.matmul(q, k.transpose(-2, -1)) * temp_scale

#     if mask is not None:
#         if mask.dim() == 2:
#             mask = mask.unsqueeze(0).unsqueeze(0)
#         elif mask.dim() == 3:
#             mask = mask.unsqueeze(1)

#         expected_mask_shape = (batch_size, num_heads, q_len, k_len)
#         if mask.shape != expected_mask_shape:
#              try:
#                  mask = mask.expand(expected_mask_shape)
#              except RuntimeError:
#                  raise ValueError(f"Mask shape {mask.shape} is not compatible with attention scores shape {expected_mask_shape}")

#         if mask.dtype == torch.bool:
#             attn_scores = attn_scores.masked_fill(mask, float("-inf"))
#         else:
#             attn_scores = attn_scores + mask

#     attn_weights = F.softmax(attn_scores, dim=-1)

#     if dropout_p > 0.0:
#         attn_weights = F.dropout(attn_weights, p=dropout_p)

#     attn_output = torch.matmul(attn_weights, v)

#     return attn_output, attn_weights

# class BaseAttention(nn.Module):
#     """Base class for attention mechanisms with common functionality."""
#     use_sdpa = True

#     def __init__(self, dims: int, head: int, max_dist: int = 512, dropout: float = 0.0):
#         """
#         Args:
#             dims: Embedding dimension.
#             head: Number of attention heads.
#             max_dist: Maximum attention distance (used by some subclasses).
#             dropout: Dropout probability for attention weights.
#         """
#         super().__init__()
#         assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
#         self.dims = dims
#         self.head = head
#         self.head_dim = dims // head
#         self.max_dist = max_dist
#         self.dropout = dropout

#     def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Reshape tensor from (batch, seq_len, dims) to
#         (batch, head, seq_len, head_dim) for multi-head attention.
#         """
#         batch, seq_len, _ = tensor.shape
#         return tensor.view(batch, seq_len, self.head, self.head_dim).transpose(1, 2).contiguous()

#     def _reshape_to_output(self, attn_output: Tensor) -> Tensor:
#         """
#         Reshape attention output from (batch, head, seq_len, head_dim)
#         back to (batch, seq_len, dims).
#         """
#         batch, _, seq_len, _ = attn_output.shape
#         return attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.dims)

# class AttentionCombiner(BaseAttention):
#     """
#     Computes attention given Q, K, V projections and applies an output projection.
#     Assumes Q, K, V inputs are already projected and appropriately shaped/scaled.
#     """
#     def __init__(self, dims: int, head: int, use_bias: bool = True, dropout: float = 0.0):
#         """
#         Args:
#             dims: Embedding dimension.
#             head: Number of attention heads.
#             use_bias: Whether to use bias in the output projection.
#             dropout: Dropout probability for attention weights.
#         """
#         super().__init__(dims, head, dropout=dropout)
#         self.out = Linear(in_features=dims, out_features=dims, bias=use_bias)
#         self._init_weights()

#     def _init_weights(self):
#         """Initialize output projection weights."""
#         nn.init.normal_(tensor=self.out.weight, std=0.02)
#         if self.out.bias is not None:
#             nn.init.zeros_(tensor=self.out.bias)

#     # @autocast('cuda', enabled=torch.cuda.is_available())
#     def forward(self, q: Tensor, k: Tensor, v: Tensor,
#                 mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
#         """
#         Processes Q, K, V through attention and output projection.

#         Args:
#             q: Query tensor (Batch, Heads, SeqLen_Q, HeadDim). Assumed scaled.
#             k: Key tensor (Batch, Heads, SeqLen_K, HeadDim). Assumed scaled.
#             v: Value tensor (Batch, Heads, SeqLen_K, HeadDim).
#             mask: Attention mask.
#             is_causal: Whether to apply causal masking (if mask is None).

#         Returns:
#             Output tensor (Batch, SeqLen_Q, Dims).
#         """
#         attn_output, _ = calculate_attention(
#             q, k, v, mask=mask,
#             temperature=1.0,
#             use_sdpa=BaseAttention.use_sdpa,
#             is_causal=is_causal,
#             dropout_p = self.dropout
#         )

#         output = self._reshape_to_output(attn_output)
#         return self.out(output)

# class AdaptiveUpdateAttention(BaseAttention):
#     """
#     Attention implementation where Key and Value representations are cached
#     and only updated based on content-dependent predictors. Suitable for
#     encoder layers or cross-attention where K/V context changes less frequently.

#     Note: Current implementation focuses on conditional update based on *current*
#     input, not standard auto-regressive KV caching for generation.
#     """
#     def __init__(self, dims: int, head: int, max_dist: int = 512, update_threshold: float = 0.5, dropout: float = 0.0):
#         """
#         Args:
#             dims: Embedding dimension.
#             head: Number of attention heads.
#             max_dist: Maximum attention distance (inherited, may not be directly used here).
#             update_threshold: Threshold for sigmoid output of predictors to trigger update.
#             dropout: Dropout probability for attention weights.
#         """
#         super().__init__(dims, head, max_dist, dropout=dropout)

#         self.query_module = ProjectionModule(dims, head, "query")
#         self.key_module = ProjectionModule(dims, head, "key")
#         self.value_module = ProjectionModule(dims, head, "value")
#         self.combiner = AttentionCombiner(dims, head, dropout=dropout)

#         self.key_update_predictor = nn.Sequential(
#             Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())
#         self.value_update_predictor = nn.Sequential(
#             Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())

#         self.update_threshold = update_threshold
#         self.stored_key_cache: Optional[Tensor] = None
#         self.stored_value_cache: Optional[Tensor] = None
#         self.reset_cache_on_forward = True

#     def _should_update(self, x: torch.Tensor, predictor: nn.Module) -> torch.Tensor:
#         """Predict whether K or V should be updated based on content."""
#         avg_rep = x.mean(dim=1)
#         update_prob = predictor(avg_rep)
#         return update_prob > self.update_threshold

#     def forward(self, x: Tensor, xa: Optional[Tensor] = None,
#                 mask: Optional[Tensor] = None,
#                 is_causal: bool = False) -> Tensor:
#         """
#         Process inputs with adaptive K/V update mechanism.

#         Args:
#             x: Input tensor for queries (Batch, SeqLen_Q, Dims).
#             xa: Optional input tensor for keys/values (for cross-attention).
#                 If None, uses x for self-attention (Batch, SeqLen_KV, Dims).
#             mask: Attention mask.
#             is_causal: Whether attention should be causal.

#         Returns:
#             Output tensor (Batch, SeqLen_Q, Dims).
#         """
#         if self.reset_cache_on_forward:
#              self.stored_key_cache = None
#              self.stored_value_cache = None

#         batch, ctx_q, _ = x.shape
#         q = self.query_module(x)

#         kv_input = xa if xa is not None else x
#         ctx_kv = kv_input.size(1)

#         update_k_batch = self._should_update(kv_input, self.key_update_predictor)
#         update_v_batch = self._should_update(kv_input, self.value_update_predictor)

#         if self.stored_key_cache is None or self.stored_key_cache.shape[2] != ctx_kv or self.stored_key_cache.shape[0] != batch:
#             k = self.key_module(kv_input)
#             self.stored_key_cache = k
#         elif update_k_batch.any():
#             new_k = self.key_module(kv_input)
#             update_mask_k = update_k_batch.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
#             k = torch.where(update_mask_k, new_k, self.stored_key_cache)
#             self.stored_key_cache = k
#         else:
#             k = self.stored_key_cache

#         if self.stored_value_cache is None or self.stored_value_cache.shape[2] != ctx_kv or self.stored_value_cache.shape[0] != batch:
#             v = self.value_module(kv_input)
#             self.stored_value_cache = v
#         elif update_v_batch.any():
#             new_v = self.value_module(kv_input)
#             update_mask_v = update_v_batch.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
#             v = torch.where(update_mask_v, new_v, self.stored_value_cache)
#             self.stored_value_cache = v
#         else:
#             v = self.stored_value_cache

#         output = self.combiner(q, k, v, mask=mask, is_causal=is_causal)
#         return output

# class Refiner:
#     """
#     Q-learning based agent to refine parameters (e.g., attention span).
#     Operates outside the standard backpropagation loop.
#     """
#     def __init__(self, states: int, actions: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
#         self.states = states
#         self.actions = actions
#         self.R = {}
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.default_value = 0.0

#     def get_value(self, state: int, action: int) -> float:
#         """Get Q-value for state-action pair."""
#         return self.R.get((state, action), self.default_value)

#     def set_value(self, state: int, action: int, value: float):
#         """Set Q-value for state-action pair."""
#         self.R[(state, action)] = value

#     def choose_action(self, state: int) -> int:
#         """Choose action using epsilon-greedy strategy."""
#         if np.random.random() < self.epsilon:
#             return np.random.randint(self.actions)
#         else:
#             action_values = [self.get_value(state, a) for a in range(self.actions)]
#             return np.argmax(action_values).item()

#     def update(self, state: int, action: int, reward: float, next_state: int):
#         """Update Q-value using the Q-learning rule."""
#         next_values = [self.get_value(next_state, a) for a in range(self.actions)]
#         best_next_value = max(next_values) if next_values else self.default_value

#         old_value = self.get_value(state, action)
#         td_target = reward + self.gamma * best_next_value
#         td_error = td_target - old_value
#         new_value = old_value + self.alpha * td_error
#         self.set_value(state, action, new_value)

# class Predictor(nn.Module):
#     """Neural predictor for estimating a scale value (e.g., for adaptive span)."""
#     def __init__(self, dims: int):
#         super().__init__()
#         self.linear = Linear(in_features=dims, out_features=1)
#         self._init_weights()

#     def _init_weights(self):
#         """Initialize predictor weights."""
#         nn.init.xavier_normal_(self.linear.weight)
#         if self.linear.bias is not None:
#            nn.init.zeros_(self.linear.bias)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Predicts a scale factor (0-1) from input features.

#         Args:
#             x: Input tensor (Batch, SeqLen, Dims) or (Batch, Dims).

#         Returns:
#             Scale tensor (Batch, 1).
#         """
#         if x.dim() > 2:
#             x = x.mean(dim=1)
#         scale = torch.sigmoid(self.linear(x))
#         return scale

# class AdaptiveSpanAttention(BaseAttention):
#     """
#     Attention mechanism where the span is dynamically adjusted based on a
#     learnable parameter or predicted scale. This version focuses on slicing
#     the input sequence to the effective span.

#     Note: This implementation attends only to the *first* `eff_span` tokens.
#     For attending to a *relative* window, different logic (e.g., sliding window
#     or masking) would be needed in `calculate_attention`.
#     """
#     def __init__(self, dims: int, head: int, max_dist: int = 512,
#                  initial_span_scale: float = 1.0, learnable_scale: bool = True,
#                  sharpen: bool = True, temp_scale: float = 0.01, dropout: float = 0.0):
#         """
#         Args:
#             dims, head, max_dist, dropout: Standard BaseAttention params.
#             initial_span_scale: Initial value for the span scale.
#             learnable_scale: If True, span_scale is an nn.Parameter.
#             sharpen, temp_scale: Parameters for dynamic temperature adjustment.
#         """
#         super().__init__(dims, head, max_dist, dropout=dropout)
#         self.sharpen = sharpen
#         self.temp_scale = temp_scale
#         if learnable_scale:
#             self.span_scale = nn.Parameter(torch.tensor(initial_span_scale))
#         else:
#             self.register_buffer("span_scale", torch.tensor(initial_span_scale))

#         self.query_module = ProjectionModule(dims, head, "query")
#         self.key_module = ProjectionModule(dims, head, "key")
#         self.value_module = ProjectionModule(dims, head, "value")
#         self.out_proj = Linear(dims, dims)

#     @autocast('cuda', enabled=torch.cuda.is_available())
#     def forward(self, x: Tensor, xa: Optional[Tensor] = None,
#                 mask: Optional[Tensor] = None,
#                 span_scale_override: Optional[Tensor] = None,
#                 is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
#         """
#         Computes attention over an adaptively determined span.

#         Args:
#             x: Input tensor for Q (Batch, SeqLen_Q, Dims).
#             xa: Optional input for K/V (Batch, SeqLen_KV, Dims). If None, use x.
#             mask: External attention mask.
#             span_scale_override: Optional tensor (Batch, 1) or scalar to override internal span_scale.
#             is_causal: Whether to apply causal masking.

#         Returns:
#             Tuple of (output tensor (Batch, SeqLen_Q, Dims), attention weights (optional)).
#         """
#         kv_input = xa if xa is not None else x
#         batch, ctx_q, _ = x.shape
#         ctx_kv = kv_input.size(1)

#         current_span_scale = span_scale_override if span_scale_override is not None else self.span_scale
#         if isinstance(current_span_scale, nn.Parameter):
#              span_scale_val = current_span_scale.sigmoid()
#         elif current_span_scale.numel() == 1:
#              span_scale_val = current_span_scale.expand(batch, 1)
#         else:
#              span_scale_val = current_span_scale

#         span_mean = span_scale_val.mean().item()
#         max_span_len = ctx_kv
#         target_span_len = max(1, int(max_span_len * span_mean))

#         eff_span = min(target_span_len, self.max_dist, ctx_q, ctx_kv)

#         if eff_span == 0:
#             return (torch.zeros_like(x), None)

#         q_span = x[:, :eff_span, :]
#         k_span = kv_input[:, :eff_span, :]
#         v_span = kv_input[:, :eff_span, :]

#         q_proj = self.query_module(q_span)
#         k_proj = self.key_module(k_span)
#         v_proj = self.value_module(v_span)

#         temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
#                        if self.sharpen
#                        else 0.5 + self.temp_scale * span_mean)
#         temperature = max(temperature, 1e-3)

#         span_mask = None
#         if mask is not None:
#              if mask.dim() == 4:
#                  span_mask = mask[:, :, :eff_span, :eff_span]
#              elif mask.dim() == 2:
#                  span_mask = mask[:eff_span, :eff_span]
#         attn_output_span, attn_weights = calculate_attention(
#             q_proj, k_proj, v_proj,
#             mask=span_mask,
#             temperature=temperature,
#             use_sdpa=BaseAttention.use_sdpa,
#             is_causal=is_causal,
#             dropout_p=self.dropout
#         )

#         output_span = self._reshape_to_output(attn_output_span)
#         projected_output_span = self.out_proj(output_span)

#         output = torch.zeros_like(x)
#         output[:, :eff_span, :] = projected_output_span

#         return output, attn_weights

# class MyelinatedLayer(BaseAttention):
#     """
#     A complex Transformer layer featuring:
#     - Integrated local/global attention (via IntegratedAttention).
#     - Optional adapters within sub-layers.
#     - Node importance prediction for sparsity.
#     - MLP block.
#     - Working memory component.
#     - Potential layer skipping ("jumping") based on a learned policy.

#     (This version assumes IntegratedAttention is the core attention mechanism).
#     """
#     def __init__(self, dims: int, head: int, num_layers: int = 3,
#                  sparsity_threshold: float = 0.1, max_dist: int = 512,
#                  dropout: float = 0.1, mlp_ratio: int = 4):
#         super().__init__(dims, head, max_dist, dropout)
#         self.num_layers = num_layers
#         self.sparsity_threshold = sparsity_threshold

#         self.attention = IntegratedAttention(dims, head, max_dist=max_dist, dropout=dropout)

#         self.sub_layers = nn.ModuleList()
#         self.node_predictors = nn.ModuleList([
#             nn.Sequential(LayerNorm(dims), Linear(dims, 1), nn.Sigmoid())
#             for _ in range(num_layers)])

#         for i in range(num_layers):
#             self.sub_layers.append(nn.ModuleDict({
#                 'ln': LayerNorm(dims),
#                 'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
#                 'adapter': Linear(dims, dims) if i % 2 == 0 else None
#             }))

#         self.policy_net = nn.Sequential(Linear(dims, 128), nn.ReLU(), Linear(128, num_layers))
#         self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))

#         n_mlp = dims * mlp_ratio
#         self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
#         self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims), nn.Dropout(dropout))
#         self.mlp_ln = LayerNorm(dims)

#         self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
#         self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

#         self.last_memory_gate_values: Optional[Tensor] = None

#     def predict_node_importance(self, x: Tensor, layer_idx: int) -> Tensor:
#         """Predict token importance mask (0.0 or 1.0) for sparsity."""
#         importance = self.node_predictors[layer_idx](x)
#         is_important = (importance > self.sparsity_threshold).float()
#         return is_important

#     def forward(self, x: Tensor, xa: Optional[Tensor] = None,
#                 mask: Optional[Tensor] = None, kv_cache: Optional[Tensor] = None,
#                 is_causal: bool = True) -> Tensor:
#         batch, ctx, _ = x.shape
#         working_memory = self.working_memory.expand(batch, 1, -1).to(x.device)
#         original_x = x

#         pooled_representation = x.mean(dim=1)
#         policy_logits = self.policy_net(pooled_representation)
#         policy = F.softmax(policy_logits, dim=-1)

#         jump_history = []
#         i = 0
#         last_processed_output = x

#         while i < self.num_layers:
#             layer = self.sub_layers[i]

#             node_importance_mask = self.predict_node_importance(x, i)

#             if node_importance_mask.mean() < 0.2 and i > 0:
#                 i += 1
#                 jump_history.append(f"skip_low_imp->{i}")
#                 continue

#             norm_x = layer['ln'](x)

#             current_attn_mask = node_importance_mask.permute(0, 2, 1)
#             if mask is not None:
#                  pass

#             attn_output = self.attention(
#                 norm_x * node_importance_mask,
#                 xa=xa,
#                 mask=mask,
#                 kv_cache=kv_cache,
#                 is_causal=is_causal
#             )

#             if layer['adapter'] is not None:
#                 attn_output = layer['adapter'](attn_output)

#             gate_value = layer['gate'](norm_x)
#             x = x + gate_value * attn_output * node_importance_mask
#             last_processed_output = x

#             memory_gate = self.memory_gate(x.mean(dim=1, keepdim=True))
#             current_mean_x = x.mean(dim=1, keepdim=True)
#             working_memory = memory_gate * working_memory + (1 - memory_gate) * current_mean_x

#             if i < self.num_layers - 1:
#                  jump_prob_dist = policy[:, 1:]
#                  jump_prob = jump_prob_dist.sum(dim=-1)

#                  should_jump_batch = torch.rand_like(jump_prob) < jump_prob

#                  if should_jump_batch.any():
#                      jump_len_probs = policy[should_jump_batch, :self.num_layers-i]
#                      sampled_jump_len = torch.multinomial(jump_len_probs, 1)[:, 0] + 1

#                      jump_length = sampled_jump_len.max().item()
#                      i_next = min(i + jump_length, self.num_layers)

#                      skip_weight_idx = min(jump_length - 1, len(self.jump_weights) - 1)
#                      skip_weight = self.jump_weights[skip_weight_idx]

#                      x = skip_weight * original_x + (1 - skip_weight) * working_memory.expand_as(x) + x * (1-skip_weight)
#                      jump_history.append(f"jump_{jump_length} S:{skip_weight.item():.2f} ->{i_next}")
#                      i = i_next
#                      continue

#             i += 1

#         mlp_input = last_processed_output
#         norm_mlp_input = self.mlp_ln(mlp_input)
#         mlp_output = self.mlp(norm_mlp_input)
#         mlp_gate_value = self.mlp_gate(norm_mlp_input)
#         final_output = mlp_input + mlp_gate_value * mlp_output

#         if 'memory_gate' in locals():
#              self.last_memory_gate_values = memory_gate.detach().clone()

#         return final_output

# class IntegratedAttention(BaseAttention):
#     """
#     Integrates multiple attention strategies:
#     - Local attention (sliding window or adaptive span via AdaptiveSpanAttention).
#     - Global attention (potentially with adaptive updates via AdaptiveUpdateAttention).
#     - Cross-attention capability.
#     - RL-based refinement (`Refiner`) of the local attention span.
#     - Iterative refinement (`_focus`) within local attention.
#     """
#     def __init__(self, dims: int, head: int, max_dist: int = 512,
#                  win_size: int = 256, max_span: int = 384, temp_scale: float = 0.01,
#                  dropout: float = 0.1,
#                  rl_states: int = 10000, rl_actions: int = 10, rl_alpha: float = 0.1,
#                  rl_gamma: float = 0.9, rl_epsilon: float = 0.1):
#         super().__init__(dims, head, max_dist, dropout=dropout)
#         self.max_span = max_span
#         self.sliding_window = win_size
#         self.temp_scale = temp_scale
#         self.sharpen = True

#         self.refiner = Refiner(
#             states=rl_states, actions=rl_actions, alpha=rl_alpha,
#             gamma=rl_gamma, epsilon=rl_epsilon)
#         self.span_pred = Predictor(dims=dims)

#         self.attn_local = AdaptiveSpanAttention(
#             dims=dims, head=head, max_dist=max_dist, sharpen=self.sharpen,
#             temp_scale=temp_scale, learnable_scale=False,
#             dropout=dropout)

#         self.attn_global = AdaptiveUpdateAttention(
#             dims=dims, head=head, max_dist=max_dist, dropout=dropout)

#         self.cross_attn = AttentionCombiner(dims=dims, head=head, dropout=dropout)

#         self.self_projection = Linear(in_features=2 * dims, out_features=dims)
#         self.global_cross_projection = Linear(in_features=dims, out_features=dims)

#         self.ln_local_in = LayerNorm(normalized_shape=dims)
#         self.ln_global_in = LayerNorm(normalized_shape=dims)
#         self.ln_cross_in = LayerNorm(normalized_shape=dims)

#         self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
#         self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

#     def forward(self, x: Tensor, xa: Optional[Tensor] = None,
#                 mask: Optional[Tensor] = None, kv_cache: Optional[Tensor] = None,
#                 is_causal: bool = True) -> Tensor:
#         """
#         Main forward pass distributing to cross or self-attention pathways.

#         Args:
#             x: Primary input tensor (Batch, SeqLen_Q, Dims).
#             xa: Context tensor for cross-attention (Batch, SeqLen_KV, Dims).
#             mask: Attention mask (padding or causal).
#             kv_cache: Key/Value cache for generation (specific usage depends on sub-modules).
#             is_causal: Flag for causal masking in self-attention.

#         Returns:
#             Output tensor (Batch, SeqLen_Q, Dims).
#         """
#         batch, ctx_q, _ = x.shape

#         if xa is not None:
#             q_norm = self.ln_cross_in(x)
#             k_cross = self.attn_global.key_module(xa)
#             v_cross = self.attn_global.value_module(xa)
#             q_cross = self.attn_global.query_module(q_norm)

#             cross_out = self.cross_attn(q=q_cross, k=k_cross, v=v_cross, mask=mask, is_causal=False)
#             return self.global_cross_projection(cross_out)

#         local_input = self.ln_local_in(x)
#         global_input = self.ln_global_in(x)

#         globe_out_raw = self.attn_global(
#              global_input,
#              xa=None,
#              mask=mask,
#              is_causal=is_causal
#         )
#         globe_out = self.global_cross_projection(globe_out_raw)

#         base_freq_scale = self.span_pred(globe_out)

#         state = self._extract_rl_state(local_input)
#         action = self.refiner.choose_action(state=state)
#         refinement_scale = self._action_to_scale(action=action)
#         final_span_scale = torch.clamp(base_freq_scale * refinement_scale.expand_as(base_freq_scale), min=0.0, max=1.0)

#         span_mean = final_span_scale.mean().item()
#         with torch.no_grad():
#             current_win_size = max(1, int(self.sliding_window * span_mean))
#             current_span_len = max(1, int(self.max_span * span_mean))
#         local_out_raw = self._slide_win_local(
#             x=local_input,
#             win_size=current_win_size,
#             span_len=current_span_len,
#             span_scale=final_span_scale,
#             mask=mask,
#             is_causal=is_causal
#         )
#         with torch.no_grad():
#              reward = self._calculate_rl_reward(output=local_out_raw)
#              next_state = self._extract_rl_state(local_out_raw)
#              self.refiner.update(state=state, action=action, reward=reward, next_state=next_state)

#         combined = torch.cat([local_out_raw, globe_out], dim=-1)
#         output = self.self_projection(combined)

#         return output

#     def _calculate_rl_reward(self, output: Tensor) -> float:
#         """Calculate quality metric (reward) for reinforcement learning."""
#         with torch.no_grad():
#             output_probs = torch.softmax(output, dim=-1)
#             safe_probs = torch.clamp(output_probs, min=1e-10)
#             entropy = -(safe_probs * torch.log(safe_probs)).sum(-1).mean()
#             coverage = (output.abs() > 0.01).float().mean()
#             reward = float(coverage - 0.1 * entropy)
#         return reward

#     def _extract_rl_state(self, x: Tensor) -> int:
#         """Extract discrete state features for RL agent from tensor."""
#         with torch.no_grad():
#             pooled = x.mean(dim=1)
#             mean_state = pooled[0].mean()
#             var_state = pooled[0].var(unbiased=False)
#             state_features = torch.stack([mean_state, var_state]).cpu().numpy()
#             state_id = self._discretize_state(state_features)
#         return state_id

#     def _discretize_state(self, state: np.ndarray) -> int:
#         """Convert continuous state numpy array to a discrete state ID."""
#         bins = np.linspace(-1, 1, num=10)
#         state_discrete = np.digitize(state, bins)
#         state_hash = sum(val * (10**i) for i, val in enumerate(state_discrete))
#         state_id = int(state_hash % self.refiner.states)
#         return state_id

#     def _action_to_scale(self, action: int) -> Tensor:
#         """Convert discrete RL action index to a continuous scale factor [0, 1]."""
#         span_value = action / (self.refiner.actions - 1)
#         scale_tensor = torch.tensor([span_value], device=self.span_pred.linear.weight.device, dtype=torch.float)
#         return scale_tensor

#     def _focus(self, query: Tensor, key: Tensor, value: Tensor,
#                span_scale: Tensor, mask: Optional[Tensor] = None,
#                is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
#         """
#         Iterative attention refinement. Applies attention multiple times,
#         adding the output back to the query. Uses manual attention calculation.

#         Args:
#             query, key, value: Input tensors (B, SeqLen_Window, D).
#             span_scale: Scale factor (scalar or B, 1) influencing effective span.
#             mask: Attention mask for the window.
#             is_causal: Apply causal masking within the window.

#         Returns:
#             Tuple (refined_output (B, SeqLen_Window, D), attention_weights (optional, None here)).
#         """
#         max_iterations = 5
#         iteration = 0
#         prev_attn_out = torch.zeros_like(query)
#         attn_out = torch.zeros_like(query)
#         threshold = self.threshold.item()
#         s_factor = self.s_factor.item()

#         q_current = query

#         while iteration < max_iterations:
#             span_mean = span_scale.mean().item()
#             target_span_len = max(1, int(self.max_span * span_mean))
#             eff_span = min(target_span_len, self.max_dist, q_current.size(1), key.size(1))

#             if eff_span == 0: break

#             q_iter = q_current[:, :eff_span, :]
#             k_iter = key[:, :eff_span, :]
#             v_iter = value[:, :eff_span, :]

#             q_proj = self.attn_local.query_module(q_iter)
#             k_proj = self.attn_local.key_module(k_iter)
#             v_proj = self.attn_local.value_module(v_iter)

#             temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
#                            if self.sharpen
#                            else 0.5 + self.temp_scale * span_mean)
#             temperature = max(temperature, 1e-3)

#             iter_mask = None
#             if mask is not None:
#                 if mask.dim() == 4: iter_mask = mask[:, :, :eff_span, :eff_span]
#                 elif mask.dim() == 2: iter_mask = mask[:eff_span, :eff_span]
#             attn_output_iter, _ = calculate_attention(
#                  q_proj, k_proj, v_proj,
#                  mask=iter_mask,
#                  temperature=temperature,
#                  use_sdpa=False,
#                  is_causal=is_causal,
#                  dropout_p=self.dropout
#             )

#             attn_out_span = self.attn_local._reshape_to_output(attn_output_iter)
#             projected_attn_out_span = self.attn_local.out_proj(attn_out_span)

#             current_iter_out = torch.zeros_like(q_current)
#             current_iter_out[:, :eff_span, :] = projected_attn_out_span

#             diff = torch.abs(current_iter_out - prev_attn_out).mean()
#             dynamic_threshold = threshold + s_factor * diff

#             if diff < dynamic_threshold and iteration > 0:
#                  attn_out = current_iter_out
#                  break

#             prev_attn_out = current_iter_out.clone()
#             q_current = q_current + current_iter_out
#             attn_out = current_iter_out

#             iteration += 1

#         return attn_out, None

#     @autocast('cuda', enabled=torch.cuda.is_available())
#     def _slide_win_local(self, x: Tensor, win_size: int, span_len: int,
#                          span_scale: Tensor, mask: Optional[Tensor] = None,
#                          is_causal: bool = False) -> Tensor:
#         """
#         Process input with sliding window attention, using `_focus` for each window.

#         Args:
#             x: Input tensor (Batch, SeqLen, Dims).
#             win_size: Size of the attention window for queries.
#             span_len: Max length of keys/values relative to query window start.
#             span_scale: Span scale tensor (Batch, 1 or scalar) passed to _focus.
#             mask: Full attention mask.
#             is_causal: Apply causal masking within windows.

#         Returns:
#             Output tensor (Batch, SeqLen, Dims).
#         """
#         batch, ctx, dims = x.size()
#         output = torch.zeros_like(x)

#         num_windows = (ctx + win_size - 1) // win_size

#         for i in range(num_windows):
#             q_start = i * win_size
#             q_end = min(q_start + win_size, ctx)
#             current_window_q_len = q_end - q_start
#             if current_window_q_len == 0: continue

#             kv_start = max(0, q_end - span_len)
#             kv_end = q_end
#             query_win = x[:, q_start:q_end, :]
#             key_win = x[:, kv_start:kv_end, :]
#             value_win = x[:, kv_start:kv_end, :]

#             window_mask = None
#             if mask is not None:
#                 if mask.dim() == 4:
#                     window_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
#                 elif mask.dim() == 2:
#                     window_mask = mask[q_start:q_end, kv_start:kv_end]
#             attn_out_win, _ = self._focus(
#                 query=query_win,
#                 key=key_win,
#                 value=value_win,
#                 span_scale=span_scale,
#                 mask=window_mask,
#                 is_causal=is_causal
#             )

#             output[:, q_start:q_end, :] = attn_out_win

#         return output

# class CTCDecoder(nn.Module):
#     def __init__(self, input_dim: int, vocab_size: int, dims: int = 256, num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.vocab_size = vocab_size
#         self.dims = dims
        
#         self.projection = nn.Linear(input_dim, dims)
#         self.lstm = nn.LSTM(dims, dims, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
#         self.output = nn.Linear(dims * 2, vocab_size + 1)  # +1 for CTC blank token
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.projection(x)  # (batch, seq_len, dims)
#         x = self.dropout(x)
#         x, _ = self.lstm(x)  # (batch, seq_len, dims * 2)
#         x = self.dropout(x)
#         logits = self.output(x)  # (batch, seq_len, vocab_size + 1)
#         return logits

# class CTCWrapper(nn.Module):
#     def __init__(self, model: Model, vocab_size: int, dims: int = 256, num_layers: int = 2):
#         super().__init__()
#         self.model = model
#         self.ctc_decoder = CTCDecoder(
#             input_dim=model.param.dims,
#             vocab_size=vocab_size,
#             dims=dims,
#             num_layers=num_layers
#         )
        
#     def forward(self, input_ids=None, pitch=None, labels=None, input_lengths=None, label_lengths=None):
#         outputs = self.model(input_ids=input_ids, pitch=pitch)
#         x = outputs["logits"]  # (batch, seq_len, vocab_size)
#         ctc_logits = self.ctc_decoder(x)  # (batch, seq_len, vocab_size + 1)
#         loss = None
#         if labels is not None and input_lengths is not None and label_lengths is not None:
#             log_probs = torch.log_softmax(ctc_logits, dim=-1)
#             log_probs = log_probs.transpose(0, 1)
            
#             loss = torch.nn.functional.ctc_loss(
#                 log_probs,
#                 labels,
#                 input_lengths,
#                 label_lengths,
#                 blank=0,
#                 reduction='mean'
#             )
        
#         return {
#             "logits": ctc_logits,
#             "loss": loss,
#             "out": x
#         }
    
#     def decode(self, logits: Tensor, input_lengths: Optional[Tensor] = None) -> List[List[int]]:
#         probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size + 1)
#         predictions = torch.argmax(probs, dim=-1)  # (batch, seq_len)
        
#         decoded_sequences = []
#         for i, pred in enumerate(predictions):
#             seq = []
#             prev_token = None
#             for j, token in enumerate(pred):
#                 if input_lengths is not None and j >= input_lengths[i]:
#                     break
#                 if token != 0 and token != prev_token:
#                     seq.append(token.item())
#                 prev_token = token
#             decoded_sequences.append(seq)
#         return decoded_sequences

#     # ctc_model = CTCWrapper(model, vocab_size=40000, dims=256, num_layers=2)

#     # outputs = ctc_model(
#     #     input_ids=input_ids,
#     #     pitch=pitch,
#     #     labels=labels,
#     #     input_lengths=input_lengths,  # Length of each audio sequence
#     #     label_lengths=label_lengths   # Length of each text sequence
#     # )

#     # loss = outputs["loss"]

#     # outputs = ctc_model(input_ids=input_ids, pitch=pitch)
#     # logits = outputs["logits"]

#     # # Decode to text
#     # decoded_sequences = ctc_model.decode(logits, input_lengths=input_lengths)
#     # ctc_model = CTCWrapper(model, vocab_size=param.vocab, dims=256, num_layers=2).to('cuda')
    
#     # print(f"CTC model parameters: {sum(p.numel() for p in ctc_model.parameters() if p.requires_grad):,}")

# # from tensorboard import program
# # log_dir = "D:/newmodel/output/logs" 
# # tb = program.TensorBoard()
# # tb.configure(argv=[None, '--logdir', log_dir])
# # url = tb.launch()
# # print(f"TensorBoard started at {url}")

# def compute_metricsB(pred, tokenizer):
#     pred_ids = pred["predictions"]
#     label_ids = pred["label_ids"]
#     if isinstance(pred_ids, tuple):
#         pred_ids = pred_ids[0]
#     else:
#         pred_ids = pred_ids
#     if pred_ids.ndim == 3:
#         pred_ids = np.argmax(pred_ids, axis=-1)
#     label_ids[label_ids == -100] = tokenizer.pad_token_id
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#     metrics = evaluate.load(path="wer")
#     wer = metrics.compute(predictions=pred_str, references=label_str)
#     return {"wer": wer}

# def train_and_evaluate(
#     model,
#     tokenizer,
#     train_loader,
#     eval_loader,
#     optimizer,
#     scheduler,
#     loss_fn,
#     max_steps=10000,
#     device="cuda",
#     accumulation_steps=1,
#     clear_cache=True,
#     log_interval=10,
#     eval_interval=100,
#     save_interval=1000,
#     checkpoint_dir="checkpoint_dir",
#     log_dir="log_dir",
# ):
#     model.to(device)
#     global_step = 0
#     scaler = torch.GradScaler()
#     writer = SummaryWriter(log_dir=log_dir)
#     train_iterator = iter(train_loader)
#     total_loss = 0
#     step_in_report = 0
#     dataset_epochs = 0

#     progress_bar = tqdm(
#         total=max_steps, desc="Training Progress", leave=True, colour="green"
#     )

#     model.train()
#     optimizer.zero_grad()

#     while global_step < max_steps:
#         try:
#             batch = next(train_iterator)
#         except StopIteration:
#             train_iterator = iter(train_loader)
#             batch = next(train_iterator)
#             dataset_epochs += 1
#             print(f"Starting dataset epoch {dataset_epochs}")

#             if step_in_report > 0:
#                 avg_loss = total_loss / step_in_report
#                 logging.info(
#                     f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}"
#                 )
#                 total_loss = 0
#                 step_in_report = 0

#         start_time = time.time()

#         input_features = batch["input_features"].to(device)
#         input_ids = batch["input_ids"].to(device)
#         labels = batch["labels"].long().to(device)

#         with torch.autocast(device_type="cuda"):
#             input_features_encoded = model.encoder(input_features)
#             decoder_output = model.decoder(input_ids, input_features_encoded)
#             logits = decoder_output.view(-1, decoder_output.size(-1))
#             active_logits = logits.view(-1, decoder_output.size(-1))
#             active_labels = labels.view(-1)
#             active_mask = active_labels != -100
#             active_logits = active_logits[active_mask]
#             active_labels = active_labels[active_mask]
#             loss = loss_fn(active_logits, active_labels)
#             # model.adjust_freq(loss=loss.item())
#         total_loss += loss.item()
#         loss = loss / accumulation_steps

#         scaler.scale(loss).backward()

#         if (global_step + 1) % accumulation_steps == 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#             if clear_cache:
#                 torch.cuda.empty_cache()

#         end_time = time.time()
#         samples_per_sec = len(batch["input_features"]) / (end_time - start_time)

#         if global_step % log_interval == 0:
#             writer.add_scalar(
#                 tag="Loss/train",
#                 scalar_value=total_loss / (global_step + 1),
#                 global_step=global_step,
#             )
#             lr = scheduler.get_last_lr()[0]
#             writer.add_scalar(
#                 tag="LearningRate", scalar_value=lr, global_step=global_step
#             )
#             writer.add_scalar(
#                 tag="SamplesPerSec",
#                 scalar_value=samples_per_sec,
#                 global_step=global_step,
#             )

#         if global_step % eval_interval == 0:
#             model.eval()
#             eval_start_time = time.time()
#             eval_loss = 0
#             all_predictions = []
#             all_labels = []
#             batch_count = 0
#             total_samples = 0

#             with torch.no_grad():
#                 for eval_batch in eval_loader:
#                     # for eval_batch in tqdm(eval_loader, desc=f"Evaluating (Step {global_step})", leave=True, colour='green'):
#                     input_features = eval_batch["input_features"].to(device)
#                     input_ids = eval_batch["input_ids"].to(device)
#                     labels = eval_batch["labels"].long().to(device)

#                     batch = input_features.size(0)
#                     total_samples += batch

#                     input_features_encoded = model.encoder(input_features)
#                     decoder_output = model.decoder(input_ids, input_features_encoded)
#                     logits = decoder_output.view(-1, decoder_output.size(-1))
#                     loss = loss_fn(logits, labels.view(-1))
#                     eval_loss += loss.item()
#                     all_predictions.extend(
#                         torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist()
#                     )
#                     all_labels.extend(labels.cpu().numpy().tolist())
#                     batch_count += 1

#             eval_time = time.time() - eval_start_time
#             loss_avg = eval_loss / batch_count if batch_count > 0 else 0
#             predictions = {
#                 "predictions": np.array(all_predictions, dtype=object),
#                 "label_ids": np.array(all_labels, dtype=object),
#             }
#             metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

#             writer.add_scalar("Loss/eval", loss_avg, global_step)
#             writer.add_scalar("WER", metrics["wer"], global_step)
#             writer.add_scalar("EvalSamples", total_samples, global_step)
#             writer.add_scalar("EvalTimeSeconds", eval_time, global_step)
#             lr = scheduler.get_last_lr()[0]

#             print(
#                 f"• STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}"
#             )

#             logging.info(
#                 f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}"
#             )
#             # scheduler.step()
#             model.train()

#         if global_step % save_interval == 0:
#             checkpoint_path = os.path.join(
#                 checkpoint_dir, f"checkpoint_step_{global_step}.pt"
#             )
#             torch.save(model.state_dict(), checkpoint_path)
#             # print(f"Model saved at step {global_step} to {checkpoint_path}")
#             logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

#         lr = scheduler.get_last_lr()[0]
#         scheduler.step()
#         global_step += 1
#         step_in_report += 1

#         avg_loss = total_loss / (global_step + 1)
#         postfix_dict = {
#             "loss": f"{avg_loss:.4f}",
#             "lr": f"{lr:.6f}",
#             "samp": f"{samples_per_sec:.1f}",
#         }
#         progress_bar.set_postfix(postfix_dict, refresh=True)
#         progress_bar.update(1)

#     final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
#     torch.save(model.state_dict(), final_model_path)
#     print(
#         f"Training completed after {global_step} steps. Final model saved to {final_model_path}"
#     )
#     writer.close()
#     progress_bar.close()

# def mainB():

#     checkpoint_dir = "./output/checkpoints"
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
#     os.makedirs(name=log_dir, exist_ok=True)

#     logging.basicConfig(
#         filename=os.path.join(log_dir, "training.log"),
#         filemode="w",
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         level=logging.INFO,
#     )

#     token = ""
#     dataset = IterableDatasetDict()
#     dataset["train"] = load_dataset(
#         path="google/fleurs",
#         name="en_us",
#         split="train",
#         streaming=True,
#         token=token,
#         trust_remote_code=True,
#     ).select_columns(column_names=["audio", "transcription"])

#     dataset["test"] = load_dataset(
#         path="google/fleurs",
#         name="en_us",
#         split="test",
#         streaming=True,
#         token=token,
#         trust_remote_code=True,
#     ).select_columns(column_names=["audio", "transcription"])

#     debug = None

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
#         decoder_start_token_id=0,
#         pad_token_id=0,
#         eos_token_id=0,
#         act="gelu",
#     )

#     model = model

#     Collator = DataCollatorB(
#         tokenizer=tokenizer,
#         audio_ctx=param.audio_ctx,
#         text_ctx=param.text_ctx,
#         mels=param.mels,
#     )

#     train_dataloader = DataLoader(
#         dataset=dataset["train"], batch_size=1, collate_fn=Collator, num_workers=0
#     )

#     eval_dataloader = DataLoader(
#         dataset=dataset["test"], batch_size=1, collate_fn=Collator, num_workers=0
#     )

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=5e-4, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.98)
#     )
#     scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=0.25, total_iters=10000, last_epoch=-1
#     )

#     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

#     train_and_evaluate(
#         model=model,
#         tokenizer=tokenizer,
#         train_loader=train_dataloader,
#         eval_loader=eval_dataloader,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         loss_fn=loss_fn,
#         max_steps=10000,
#         device="cuda",
#         accumulation_steps=1,
#         clear_cache=False,
#         log_interval=10,
#         eval_interval=500,
#         save_interval=10000,
#         checkpoint_dir=checkpoint_dir,
#         log_dir=log_dir,
#     )

# def train_and_evaluate(
#     model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn,
#     max_steps=10000, device='cuda', accumulation_steps=1, clear_cache=True,
#     log_interval=10, eval_interval=100, save_interval=1000,
#     checkpoint_dir="checkpoint_dir", log_dir="log_dir"
# ):
#     model.to(device)
#     global_step = 0
#     scaler = torch.GradScaler()
#     writer = SummaryWriter(log_dir=log_dir)
#     train_iterator = iter(train_loader)
#     total_loss = 0
#     step_in_report = 0
#     dataset_epochs = 0

#     progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour='green')

#     model.train()
#     optimizer.zero_grad()

#     while global_step < max_steps:
#         try:
#             batch = next(train_iterator)
#         except StopIteration:
#             train_iterator = iter(train_loader)
#             batch = next(train_iterator)
#             dataset_epochs += 1
#             print(f"Starting dataset epoch {dataset_epochs}")

#             if step_in_report > 0:
#                 avg_loss = total_loss / step_in_report
#                 logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
#                 total_loss = 0
#                 step_in_report = 0

#         start_time = time.time()

#         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

#         with torch.autocast(device_type="cuda"):
#             output = model(**batch) if hasattr(model, '__call__') else model.forward(**batch)
#             logits = output["logits"] if isinstance(output, dict) and "logits" in output else output
#             labels = batch["labels"]
#             active_logits = logits.view(-1, logits.size(-1))
#             active_labels = labels.view(-1)
#             active_mask = active_labels != 0
#             active_logits = active_logits[active_mask]
#             active_labels = active_labels[active_mask]
#             loss = loss_fn(active_logits, active_labels)
#         total_loss += loss.item()
#         loss = loss / accumulation_steps

#         scaler.scale(loss).backward()

#         if (global_step + 1) % accumulation_steps == 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             if clear_cache:
#                 torch.cuda.empty_cache()

#         end_time = time.time()
#         samples_per_sec = batch["spectrogram"].size(0) / (end_time - start_time)

#         if global_step % log_interval == 0:
#             writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
#             lr = scheduler.get_last_lr()[0]
#             writer.add_scalar(tag='LearningRate', scalar_value=lr, global_step=global_step)
#             writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)

#         if global_step % eval_interval == 0:
#             model.eval()
#             eval_start_time = time.time()
#             eval_loss = 0
#             all_predictions = []
#             all_labels = []
#             batch_count = 0
#             total_samples = 0

#             with torch.no_grad():
#                 for eval_batch in eval_loader:
#                     eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
#                     output = model(**eval_batch) if hasattr(model, '__call__') else model.forward(**eval_batch)
#                     logits = output["logits"] if isinstance(output, dict) and "logits" in output else output
#                     labels = eval_batch["labels"]
#                     batch_size = logits.size(0)
#                     total_samples += batch_size
#                     loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
#                     eval_loss += loss.item()
#                     all_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
#                     all_labels.extend(labels.cpu().numpy().tolist())
#                     batch_count += 1

#             eval_time = time.time() - eval_start_time
#             loss_avg = eval_loss / batch_count if batch_count > 0 else 0
#             predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
#             metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

#             writer.add_scalar('Loss/eval', loss_avg, global_step)
#             writer.add_scalar('WER', metrics['wer'], global_step)
#             writer.add_scalar('EvalSamples', total_samples, global_step)
#             writer.add_scalar('EvalTimeSeconds', eval_time, global_step)

#             lr = scheduler.get_last_lr()[0]
#             print(f"• STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")
#             logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
#             model.train()

#         if global_step % save_interval == 0:
#             checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
#             torch.save(model.state_dict(), checkpoint_path)
#             logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

#         lr = scheduler.get_last_lr()[0]
#         scheduler.step()
#         global_step += 1
#         step_in_report += 1

#         avg_loss = total_loss / (global_step + 1)
#         postfix_dict = {
#             'loss': f'{avg_loss:.4f}',
#             'lr': f'{lr:.6f}',
#             'samp': f'{samples_per_sec:.1f}'
#         }
#         progress_bar.set_postfix(postfix_dict, refresh=True)
#         progress_bar.update(1)

#     final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
#     torch.save(model.state_dict(), final_model_path)
#     print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
#     writer.close()
#     progress_bar.close()

# def get_optimizer(model, lr=5e-4, weight_decay=0.01):
#     return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6, betas=(0.9, 0.98))

# def get_scheduler(optimizer, total_steps=10000):
#     return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=total_steps, last_epoch=-1)

# def get_loss_fn():
#     return torch.nn.CrossEntropyLoss(ignore_index=0)

# def mainc():
#     token = ""
#     log_dir = os.path.join('./output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
#     os.makedirs(log_dir, exist_ok=True)
#     tokenizer = setup_tokenizer(token)

#     param = Dimensions(
#         mels=128, aud_ctx=1500, aud_head=4, aud_dims=512, aud_idx=4,
#         vocab=40000, text_ctx=512, text_head=4, text_dims=512, text_idx=4,
#         act="swish", debug={}, cross_attn=True, features=["spectrogram"]
#     )

#     dataset_config = {
#         "spectrogram": True, "waveforms": False, "pitch": False, "downsamples": False,
#         "frequency": True, "hilbert": False, "hop_length": 128, "fmin": 150, "fmax": 2000,
#         "n_mels": 128, "n_fft": 1024, "sampling_rate": 16000, "pad_mode": "constant",
#         "center": True, "power": 2.0, "window_fn": torch.hann_window, "mel_scale": "htk",
#         "norm": None, "normalized": False
#     }

#     model = create_model(param)
#     train_dataset, test_dataset = prepare_datasets(
#         tokenizer=tokenizer, token=token, sanity_check=False, dataset_config=dataset_config
#     )

#     collator = DataCollator(tokenizer=tokenizer)
#     train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collator, num_workers=0)
#     eval_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator, num_workers=0)

#     optimizer = get_optimizer(model)
#     scheduler = get_scheduler(optimizer)
#     loss_fn = get_loss_fn()

#     train_and_evaluate(
#         model=model,
#         tokenizer=tokenizer,
#         train_loader=train_loader,
#         eval_loader=eval_loader,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         loss_fn=loss_fn,
#         max_steps=10000,
#         device='cuda',
#         accumulation_steps=1,
#         clear_cache=False,
#         log_interval=10,
#         eval_interval=500,
#         save_interval=10000,
#         checkpoint_dir="./checkpoints",
#         log_dir=log_dir
#     )

# class attention(nn.Module):
#     def __init__(self, dims: int, head: int):
#         super(attention, self).__init__()
#         self.dims = dims
#         self.head = head
#         self.head_dim = dims // head
#         self.q = nn.Linear(dims, dims)
#         self.k = nn.Linear(dims, dims, bias=False)
#         self.v = nn.Linear(dims, dims)
#         self.o = nn.Linear(dims, dims)

#         self.lna = nn.LayerNorm(dims, bias = False)
#         self.lnb = nn.LayerNorm(dims, bias = False)      
#         self.lnc = nn.LayerNorm(self.head_dim, bias = False)
#         self.lnd = nn.LayerNorm(self.head_dim, bias = False)     

#     def _forward(self, x: Tensor, xa = None, mask = None):
#         q = self.q(self.lna(x))
#         k = self.k(self.lnb(x if xa is None else xa))
#         v = self.v(self.lnb(x if xa is None else xa))
#         query = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
#         key = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
#         value = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

#         max_iterations = 5
#         iteration = 0
#         prev_attn_out = torch.zeros_like(query)
#         attn_out = torch.zeros_like(query)
#         threshold = self.threshold.item()
#         s_factor = self.s_factor.item()

#         q_current = query

#         while iteration < max_iterations:
 
#             eff_span = min(x.shape[1], xa.shape[1], q_current.size(1), key.size(1))

#             if eff_span == 0: break

#             q_iter = q_current[:, :eff_span, :]
#             k_iter = key[:, :eff_span, :]
#             v_iter = value[:, :eff_span, :]

#             q_proj = self.attn_local.query_module(q_iter)
#             k_proj = self.attn_local.key_module(k_iter)
#             v_proj = self.attn_local.value_module(v_iter)

#             temperature = (1.0 + self.temp_scale * (1.0 - xa.mean())
#                            if self.sharpen
#                            else 0.5 + self.temp_scale * xa.mean())
#             temperature = max(temperature, 1e-3)

#             iter_mask = None
#             if mask is not None:
#                 if mask.dim() == 4: iter_mask = mask[:, :, :eff_span, :eff_span]
#                 elif mask.dim() == 2: iter_mask = mask[:eff_span, :eff_span]

#             attn_output_iter, _ = calculate_attention(
#                  q_proj, k_proj, v_proj,
#                  mask=iter_mask,
#                  temperature=temperature,
#                  use_sdpa=False,
#                  dropout_p=self.dropout
#             )

#             attn_out_span = self.attn_local._reshape_to_output(attn_output_iter)
#             projected_attn_out_span = self.attn_local.out_proj(attn_out_span)

#             current_iter_out = torch.zeros_like(q_current)
#             current_iter_out[:, :eff_span, :] = projected_attn_out_span

#             diff = torch.abs(current_iter_out - prev_attn_out).mean()
#             dynamic_threshold = threshold + s_factor * diff

#             if diff < dynamic_threshold and iteration > 0:
#                  attn_out = current_iter_out
#                  break

#             prev_attn_out = current_iter_out.clone()
#             q_current = q_current + current_iter_out
#             attn_out = current_iter_out

#             iteration += 1

#         return attn_out, None

#     def _slide_win_local(self, x: Tensor, win_size: int, span_len: int,
#                          span_scale: Tensor, mask: Optional[Tensor] = None,
#                          is_causal: bool = False) -> Tensor:
#         """
#         Process input with sliding window attention, using `_focus` for each window.

#         Args:
#             x: Input tensor (Batch, SeqLen, Dims).
#             win_size: Size of the attention window for queries.
#             span_len: Max length of keys/values relative to query window start.
#             span_scale: Span scale tensor (Batch, 1 or scalar) passed to _focus.
#             mask: Full attention mask.
#             is_causal: Apply causal masking within windows.

#         Returns:
#             Output tensor (Batch, SeqLen, Dims).
#         """
#         batch, ctx, dims = x.size()
#         output = torch.zeros_like(x)

#         num_windows = (ctx + win_size - 1) // win_size

#         for i in range(num_windows):
#             q_start = i * win_size
#             q_end = min(q_start + win_size, ctx)
#             current_window_q_len = q_end - q_start
#             if current_window_q_len == 0: continue

#             kv_start = max(0, q_end - span_len)
#             kv_end = q_end
#             query_win = x[:, q_start:q_end, :]
#             key_win = x[:, kv_start:kv_end, :]
#             value_win = x[:, kv_start:kv_end, :]

#             window_mask = None
#             if mask is not None:
#                 if mask.dim() == 4:
#                     window_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
#                 elif mask.dim() == 2:
#                     window_mask = mask[q_start:q_end, kv_start:kv_end]

#             attn_out_win, _ = self._focus(
#                 query=query_win,
#                 key=key_win,
#                 value=value_win,
#                 span_scale=span_scale,
#                 mask=window_mask,
#                 is_causal=is_causal
#             )

#             output[:, q_start:q_end, :] = attn_out_win

#         return output
