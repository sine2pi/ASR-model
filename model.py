import os, torch, numpy as np
from torch.nn.functional import scaled_dot_product_attention as SDPA
from torch import nn, Tensor, einsum
from typing import Iterable
from functools import partial
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from optimizer import MaxFactor as MF
from torch.utils.data import DataLoader    
from datetime import datetime
from essentials import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

THETA = 30000.0
PATH = './cache'

@dataclass
class Dimensions:
    tokens: int
    mels: int
    dims: int
    head: int
    layer: int
    act: str
    n_type: str

class rotary(nn.Module):
    def __init__(n, dims, head):
        super().__init__()

        n.head_dim = dims // head
        n.head = head
        n.dims = dims
        n.lin = nn.Linear(dims, n.head_dim // 2, bias=True)

    def _compute_freqs(n, x=None, mask=None):
        if mask is None:
            scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype, requires_grad=False) / 2595) - 1
            return x.mean(dim=-1) * scale / 1000 if x is not None else 200 * scale / 1000
        else: 
            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype, requires_grad=False) / n.head_dim * torch.log(torch.tensor(x.mean(dim=-1) * THETA if x is not None else THETA, requires_grad=False))

    def forward(n, x=None, xa=None, mask=None): 
        t = torch.arange(x.shape[2], device=device, dtype=dtype, requires_grad=False).float()
        freqs = torch.einsum('i,j->ij', t,  n._compute_freqs(mask=mask))
        if xa is not None:
            freqs = torch.polar(xa.mean(dim=-1).unsqueeze(-1), freqs)
        else:   
            freqs = torch.polar(torch.ones_like(freqs, requires_grad=False), freqs)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        out = torch.cat([x1.type_as(x), x2], dim=-1)
        return out
    
class attention(nn.Module):
    def __init__(n, dims, head, layer, n_type=None, modal=False): 
        super().__init__()
        n.layer = layer

        n.scale = (dims // head) ** -0.25
        n.modal = modal

        n.q   = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims))

        n.conv = nn.Conv2d(head, head, 1, bias=False) if modal else nn.Identity()
        n.ln = get_norm(n_type, dims // head)
        n.rot = rotary(dims, head)

    def forward(n, x, xa=None, mask=None, pt=None, skip=False, pattern=None): 
 
        b, c, d = x.shape
        p = pattern if pattern is not None else None
        k, v = n.kv(x if xa is None else xa)
        q = n.q(x)
        q, k = n.rot(q, xa=x, mask=mask), n.rot(k, xa=x if xa is None else xa, mask=mask)  

        if isinstance(skip, int) and not have(p):
            stride = max(1, n.layer - skip)
            a = SDPA(n.ln(q), n.ln(k[:, :, ::stride, :]), v[:, :, ::stride, :], is_causal=have(mask))
            
        elif have(p) and p > 1: 
            k, v = k[:, :, ::p, :], v[:, :, ::p, :]
            a = SDPA(n.ln(q), n.ln(k), v, is_causal=have(mask))
        else: 
            a = SDPA(n.ln(q), n.ln(k), v, is_causal=have(mask))

        if n.modal and xa is not None:

            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb = n.q(x), n.q(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)

            if have(p) and p > 1:
                k, ka, kb = k[:, :, ::p, :], k[:, :, 1::p, :], k[:, :, 2::p, :]
                v, va, vb = v[:, :, ::p, :], v[:, :, 1::p, :], v[:, :, 2::p, :]
            elif isinstance(skip, int):
                stride = max(1, n.layer - skip)
                ka, va = ka[:, :, ::stride, :], va[:, :, ::stride, :]
                kb, vb = kb[:, :, ::stride, :], vb[:, :, ::stride, :]
            else:
                ka, va = ka, va
                kb, vb = kb, vb
                
            b = SDPA(n.ln(qa), n.ln(kb), vb, is_causal=have(mask))
            c = SDPA(n.ln(qb), n.ln(ka), va, is_causal=have(mask))

            return n.out(a), n.out(n.conv(b)), n.out(n.conv(c))
        else:
            return n.out(a)

class attentionb(nn.Module):
    def __init__(n, dims: int, head: int, layer: int, n_type):
        super().__init__()

        n.q   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.c   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        
        n.ln = get_norm(n_type, dims // head)
        
    def forward(n, x, xa=None, mask=None, pt=None, context_window=3):
        q = n.q(x)
        k, v = n.kv(aorb(xa, x))
        b, h, c, d = q.shape 
        scale = d ** -0.5

        if pt is not None: c = n.c(pt)
        else: c = torch.zeros_like(x, requires_grad=False)

        triplet_scores = torch.zeros(b, h, c, c, device=device, requires_grad=False)

        for i in range(c):
            for j in range(c):
                context_start = max(0, min(i, j) - context_window)
                context_end = min(c, max(i, j) + context_window)
                
                for k in range(context_start, context_end): 
                    score = (q[:, :, i, :] * k[:, :, j, :] * c[:, :, k, :]).sum(dim=-1)
                    triplet_scores[:, :, i, j] += score

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale + triplet_scores
        if have(mask): qk = qk + mask[:c, :c]
        qk = torch.nn.functional.softmax(qk, dim=-1)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        return n.out(wv)

class gate(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.features = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.top = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x, num=None):
        types, indices = torch.topk(n.top(x), num, dim=-1)
        type = torch.zeros_like(n.features(x))
        type.scatter_(-1, indices, torch.nn.functional.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, num_types=2):
        super().__init__()

        n.expand = head
        n.ln = get_norm(n_type=n_type, dims=dims)
        n.act_fn = get_activation(act)
        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        
        n.attn = attention(dims, head, layer, n_type=n_type)
        n.gate = gate(dims, num_types=n.expand) 

        n.mlp = nn.Sequential(n.ln, nn.Linear(dims, dims*n.expand), get_activation(act), nn.Linear(dims*n.expand, dims))
        n.ga = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.cs = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))        

    def forward(n, x, xa=None, mask=None, pt=None, **kwargs):
        x = x + n.attn(n.ln(x), mask=mask, pt=pt, **kwargs)
        if xa is not None: 
            xa = xa + n.audio(xa.shape[1], xa.shape[-1]).to(device, dtype)
            xa = xa + n.gate(xa, n.expand // 2) 
            xa = xa + torch.sum(torch.stack([g(xa) for g in n.ga], dim=-1) * n.cs(xa).unsqueeze(2), dim=-1)
            x = x + n.attn(n.ln(x), xa=xa, pt=pt, **kwargs)
        return x + n.mlp(x)

class attn_pass(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, skip=True, pattern=None):
        super().__init__()
        
        n.layers = nn.ModuleList()
        for i in range(layer): n.layers.append(residual(dims, head, layer, act, n_type))

    def forward(n, x, override=None):
        for i, layer in enumerate(n.layers): x = layer(x, skip=i, pattern=override[i] if override else None)
        return x

class processor(nn.Module):
    def __init__(n, tokens, mels, dims, head, layer, act, n_type, ctx=2048): 
        super().__init__()
        
        n.ln = get_norm(n_type, dims)

        n.token = nn.Embedding(tokens, dims) 
        n.position = nn.Parameter(torch.ones(ctx, dims), requires_grad=True)
        n.blend = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        n.block: Iterable[residual] = nn.ModuleList(
            [residual(dims, head, layer, act, n_type) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, seq=False) -> Tensor:

        xa, xb, xc, pt = (xa.pop(k, None) for k in ('a', 'b', 'c', 'pt')) if isinstance(xa, TensorDict) else (None, None, None, None)

        blend = torch.sigmoid(n.blend)
        x = (n.token(x) + n.position[:x.shape[-1]]).to(device, dtype)

        for i in n.block:
            a = i(x, mask=n.mask, pt=pt)
            b = i(a, xa=i(xa, pt=pt))
            c = i(b, xa=i(xb, pt=pt))
            d = i(c, xa=i(xc, pt=pt))

            for j in [(xa), (xb), (xc)]: e = i(x, xa=i(j, pt=pt))

            f = torch.cat([d, e], dim=1)
            g = i(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:])
            x = g if seq else blend * (d) + (1 - blend) * g 

        return (n.ln(x) @ torch.transpose(n.token.weight.to(dtype), 0, 1)).float()

class Model(nn.Module):
    def __init__(n, param: Dimensions):
        super().__init__()

        n.param = param
        n.processor = processor(
            tokens=param.tokens,
            mels=param.mels,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            act=param.act,
            n_type=param.n_type,
            )

        n.enc = AudioEncoder(param.mels, param.dims, param.head, param.act, param.n_type, norm=False, enc=False)

        n.layer = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer += 1        

    def forward(n, labels=None, text_ids=None, spectrogram=None, pitch=None, waveform=None, pitch_tokens=None):

        fx = next((t for t in (pitch, spectrogram, waveform) if t is not None), None)

        xa = TensorDict({
            'a': aborc(pitch, spectrogram, waveform),
            'b': aborc(spectrogram, pitch, waveform),
            'c': aborc(waveform, pitch, spectrogram),
            }, batch_size=fx.shape[0])

        x = text_ids
        xa = n.enc(no_none(xa))
        xa['pt'] = pitch_tokens if pitch_tokens is not None else None

        output = n.processor(x, xa, seq=False)

        loss = None
        if labels is not None: 
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": output, "loss": loss}

    def _init_w(n, m):
        n.counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0, "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, m in n.named_modules():
            if isinstance(m, nn.RMSNorm):
                n.counts["RMSNorm"] += 1
            if isinstance(m, nn.LayerNorm):
                n.counts["LayerNorm"] += 1                
            elif isinstance(m, nn.Linear):
                n.counts["Linear"] += 1

    def init_w(n):
        print("Initializing model w...")
        n.apply(n._init_w)
        print("Initialization summary:")
        for module_type, count in n.counts.items():
            if count > 0:
                print(f"{module_type}: {count}")
