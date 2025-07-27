import os
import warnings
import logging
from itertools import chain
import torch
from torch import nn, Tensor
from typing import Optional, Dict
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from torch.nn.functional import scaled_dot_product_attention

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

class rotary(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.theta = nn.Parameter((torch.tensor(10000, device=device, dtype=dtype)), requires_grad=True)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x, ctx) -> Tensor:
        freqs = (self.theta / 220.0) * self.freqs_base
        pos = torch.arange(ctx, device=device, dtype=dtype) 
        freqs = pos[:, None] * freqs
        freqs=torch.polar(torch.ones_like(freqs), freqs)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

def qkvinit(dims: int, head: int):
    head_dim = dims // head
    scale = head_dim ** -0.5
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    return q, k, v, o, scale

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

def calculate_attention(q, k, v, mask=None, temperature=1.0, is_causal=True):
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5

    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    # out = scaled_dot_product_attention(scaled_q, k, v, attn_mask=attn_mask, is_causal=is_causal if attn_mask is None else False)
    return out

class LocalAttentionModule(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.query_module = nn.Linear(head_dim, head_dim)
        self.key_module = nn.Linear(head_dim, head_dim)
        self.value_module = nn.Linear(head_dim, head_dim)
        self.out_proj = nn.Linear(head_dim, head_dim)
    
    def _reshape_to_output(self, x):
        return x

class attentiona(nn.Module):
    def __init__(self, dims: int, head: int, max_iterations: int = 3, threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1):
        super(attentiona, self).__init__()
        # self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_iterations = max_iterations
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.factor = nn.Parameter(torch.tensor(factor))
        self.dropout = dropout
        
        self.q = nn.Linear(dims, dims)
        self.k = nn.Linear(dims, dims, bias=False)
        self.v = nn.Linear(dims, dims)
        self.o = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims, bias=False)
        self.lnb = nn.LayerNorm(dims, bias=False)      
        self.lnc = nn.LayerNorm(self.head_dim, bias=False)
        self.lnd = nn.LayerNorm(self.head_dim, bias=False)     
        self.attn_local = LocalAttentionModule(self.head_dim)

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        q = self.q(self.lna(x))
        k = self.k(self.lnb(x if xa is None else xa))
        v = self.v(self.lnb(x if xa is None else xa))
        query = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        key = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        value = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        iteration = 0
        prev_out = torch.zeros_like(query)
        attn_out = torch.zeros_like(query)
        threshold = self.threshold.item()
        factor = self.factor.item()
        qcur = query

        while iteration < self.max_iterations:
            eff_span = min(x.shape[1], qcur.size(1), key.size(1))
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])
            if eff_span == 0: 
                break

            qiter = qcur[:, :, :eff_span, :]
            kiter = key[:, :, :eff_span, :]
            viter = value[:, :, :eff_span, :]
            q = self.attn_local.query_module(qiter)
            k = self.attn_local.key_module(kiter)
            v = self.attn_local.value_module(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_iter = calculate_attention(
                self.lnc(q), self.lnd(k), v,
                mask=iter_mask,
                is_causal=True)

            iter_out = torch.zeros_like(qcur)
            iter_out[:, :, :eff_span, :] = attn_iter
            diff = torch.abs(iter_out - prev_out).mean()
            dthresh = threshold + factor * diff

            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            qcur = qcur + iter_out
            attn_out = iter_out
            iteration += 1

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int, mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:

        batch, ctx, dims = x.size()
        output = torch.zeros_like(x)
        num_win = (ctx + win_size - 1) // win_size

        for i in range(num_win):
            qstart = i * win_size
            qend = min(qstart + win_size, ctx)
            current_win_qlen = qend - qstart
            if current_win_qlen == 0: 
                continue

            kvstart = max(0, qend - span_len)
            kvend = qend
            qwin = x[:, qstart:qend, :]
            kwin = x[:, kvstart:kvend, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, qstart:qend, kvstart:kvend]
                elif mask.dim() == 2:
                    win_mask = mask[qstart:qend, kvstart:kvend]

            attn_out, _ = self._focus(
                x=qwin,
                xa=kwin,
                mask=win_mask)
            output[:, qstart:qend, :] = attn_out
        return output

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
                use_sliding_win: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
        if use_sliding_win:
            return self._slide_win_local(x, win_size, span_len, mask)
        else:
            output, _ = self._focus(x, xa, mask)
            return output

class attentionb(nn.Module):
    def __init__(self, dims: int, head: int):
        super(attentionb, self).__init__()
        self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.rope = rotary(dims=dims, head=head)    

    def forward(self, x: Tensor, xa = None, mask = None):
        z = default(xa, x)
        q, k, v = create_qkv(self.dims, self.head, self.q, self.k, self.v, self.lna(x), self.lna(z))      
        q = self.rope(q, q.shape[2])
        k = self.rope(k, k.shape[2]) 
        a = scaled_dot_product_attention(self.lnb(q), self.lnb(k), v, is_causal=mask is not None and q.shape[1] > 1)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(out)

class Residual(nn.Module): 
    def __init__(self, dims: int, head: int, act: str = "silu"):
        super().__init__()

        self.lna = nn.LayerNorm(dims, bias=False)  
        self.attnb = attentionb(dims, head)
        self.attna = attentiona(dims, head, max_iterations=3)
        self.mlp = nn.Sequential(Linear(dims, dims*4), get_activation(act), Linear(dims*4, dims))

    def forward(self, x, xa = None, mask = None) -> Tensor:   
        x = x + self.attnb(self.lna(x), xa=None, mask=mask)
        if xa is not None:
            x = x + self.attna(self.lna(x), xa, mask=None, use_sliding_win=True, win_size=500, span_len=1500)  
        x = x + self.mlp(self.lna(x))
        return x

class processor(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, act: str = "gelu"): 
        super(processor, self).__init__()

        self.ln = nn.LayerNorm(dims)
        self.blend = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.token_emb = nn.Embedding(vocab, dims)
        self.positions = nn.Parameter(torch.empty(ctx, dims), requires_grad=True)
        self.audio_emb = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)

        act_fn = get_activation(act)        
        self.audio_enc = nn.Sequential(
            Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        self.bA = nn.ModuleList([Residual(dims, head, act_fn) for _ in range(layer)])

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, sequential=False) -> Tensor:    

        x  = self.token_emb(x.long()) + self.positions[:x.shape[1]]    
        xa = self.audio_enc(xa).permute(0, 2, 1)
        xa = xa + self.audio_emb(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        for b in chain(self.bA or []):
            xa = b(x=xa, xa=None, mask=None)
            x  = b(x=x, xa=None, mask=self.mask)
            x  = b(x=x, xa=xa, mask=None)
            # xc = b(torch.cat([x, xa], dim=1), xa=None, mask=self.mask)    
            # x  = b(x=xc[:, :x.shape[1]], xa=xc[:, x.shape[1]:], mask=None)

            # if sequential:
            #     x = y
            # else:
            #     a = torch.sigmoid(self.blend)
            #     x = a * y + (1 - a) * x 

        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.ln(x)        
        x = x @ torch.transpose(self.token_emb.weight.to(dtype), 0, 1).float()
        return x

    def init_weights(self):
        print("Initializing model weights...")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")
   
class Model(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        self.processor = processor(
            vocab=param.vocab,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            act=param.act)       
        
    def forward(self,
        labels=None, input_ids=None, pitch: Optional[torch.Tensor]=None) -> Dict[str, Optional[torch.Tensor]]:
        x = input_ids
        xa = pitch if pitch is not None else torch.zeros(1, 1, self.param.mels, device=device, dtype=dtype)
        logits = self.processor(x, xa)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return {"logits": logits, "loss": loss} 

    def _init_weights(self, module):
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
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
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
            elif isinstance(module, processor):
                self.init_counts["processor"] += 1

    def init_weights(self):
        print("Initializing model weights...")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")
