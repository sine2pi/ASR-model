
import warnings
import logging
from itertools import chain
import torch
from torch import nn, Tensor, einsum
from typing import Optional
import numpy as np
from dataclasses import dataclass
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

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

def there_is_a(val):
    return val is not None

@dataclass
class Dimensions:
    vocab: int
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str

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

    def forward(self, x) -> Tensor:
        freqs = (self.theta / 220.0) * self.freqs_base 

        pos = torch.arange(x.shape[2], device=device, dtype=dtype) 
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

def calculate_attention(q, k, v, mask=None, temp=1.0):
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
        print(temp)
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)        
    return out

class LocalOut(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head_dim = dims // head
        self.dims = dims
        self.q_hd = nn.Linear(self.head_dim, self.head_dim)
        self.k_hd = nn.Linear(self.head_dim, self.head_dim)
        self.v_hd = nn.Linear(self.head_dim, self.head_dim)
        self.out = nn.Linear(self.head_dim, self.head_dim)

    def _reshape_to_output(self, attn_output: Tensor) -> Tensor:
        batch, _, ctx, _ = attn_output.shape
        return attn_output.transpose(1, 2).contiguous().view(batch, ctx, self.dims)      

class attentionb(nn.Module):
    def __init__(self, dims: int, head: int, max_iter: int = 3, 
    threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1, temp = 1.0):
        super(attentionb, self).__init__()

        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.win = 0

        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(self.head_dim) 
        self.rope = rotary(dims, head) 

        self.max_iter = max_iter
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
        self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        
        self.factor = nn.Parameter(torch.tensor(factor), requires_grad=True)
        self.local = LocalOut(dims, head)   

    def update_win(self, win_size=None):
        if win_size is not None:
            self.win_size = win_size
            return win_size
        elif hasattr(self, 'win_size') and self.win_size is not None:
            win_size = self.win_size
            return win_size
        return None

    def _focus(self, x, xa = None, mask = None, win_size=None):

        q = self.que(self.lna(x))
        k, v = self.kv(self.lna(x if not xa else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
      
        self.scale = q.shape[-1] ** -0.35
        q = self.rope(q)
        k = self.rope(k)

        iteration = 0
        temp = self.temp
        prev_out = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = self.threshold
        factor = self.factor
        curq = q #if curq is None else curq
        
        while iteration < self.max_iter:
            eff_span = min(curq.shape[1], k.shape[1])
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])
            if eff_span == 0: 
                break

            qiter = curq[:, :, :eff_span, :]
            kiter = k[:, :, :eff_span, :]
            viter = v[:, :, :eff_span, :]
            q = self.local.q_hd(qiter)
            k = self.local.k_hd(kiter)
            v = self.local.v_hd(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_iter = calculate_attention(
                self.lnb(q), self.lnb(k), v,
                mask=iter_mask, temp=temp)

            iter_out = torch.zeros_like(curq)
            iter_out[:, :, :eff_span, :] = attn_iter
            diff = torch.abs(iter_out - prev_out).mean()
            dthresh = threshold + factor * diff
            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            curq = curq + iter_out
            attn_out = iter_out
            # if win_size is not None:
            #     if win_size > self.win:
            #         temp += 0.005
            #     else:
            #         temp -= 0.005
            #     self.win = win_size   
            iteration += 1

        out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out

    def _slide_win_local(self, x, mask = None) -> Tensor:

        win = self.update_win()
        win_size = win if win is not None else self.head_dim
        span_len = win_size + win_size // self.head

        _, ctx, _ = x.shape
        out = torch.zeros_like(x)
        windows = (ctx + win_size - 1) // win_size

        for i in range(windows):
            qstart = i * win_size
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

            attn_out = self._focus(x=qwin, xa=kwin, mask=win_mask, win_size=win_size)
            out[:, qstart:qend, :] = attn_out
        return out

    def forward(self, x, xa = None, mask = None):
            x = self._slide_win_local(x, mask=None)
            xa = self._slide_win_local(xa, mask=None)
            output = self._focus(x, xa, mask=None)
            return self.out(output)
       
class attentiona(nn.Module):
    def __init__(self, dims: int, head: int, dropout_rate: float = 0.1, cross_talk=False):
        super().__init__()
        self.head = head
        self.dims = dims
        self.cross_talk = cross_talk

        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)

        self.ln = nn.LayerNorm(dims) 
        self.rope = rotary(dims, head) 

        self.x = nn.Conv2d(head, head, 1, bias = False) if cross_talk else None
        self.xa = nn.Conv2d(head, head, 1, bias = False) if cross_talk else None

    def forward(self, x, xa = None, mask = None):

        q = self.que(self.ln(x))
        k, v = self.kv(self.ln(x if xa is None else xa)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        q = self.rope(q)
        k = self.rope(k)

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale 

        if there_is_a(mask):
            i, j = qk.shape[-2:]
            mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            qk = qk.masked_fill(mask,  -torch.finfo(qk.dtype).max)

        qk = torch.nn.functional.softmax(qk, dim=-1)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)

class attentiond(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims

        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)

        self.ln = nn.LayerNorm(dims) 
        self.rope = rotary(dims, head) 

        self.x = nn.Conv2d(head, head, 1, bias = False)
        self.xa = nn.Conv2d(head, head, 1, bias = False) 

    def forward(self, x, xa = None, mask = None):

        qk, v = self.kv(self.ln(x)).chunk(2, dim=-1) 
        qka, va = self.kv(self.ln(x if xa is None else xa)).chunk(2, dim=-1)
        qk, qka, v, va = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.head), (qk, qka, v, va))
        qk = einsum('b h i d, b h j d -> b h i j', qk, qka)
        if there_is_a(mask):
            i, j = qk.shape[-2:]
            mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
            qk = qk.masked_fill(mask, -torch.finfo(qk.dtype).max)
        x = qk.softmax(dim = -1)
        xa = qk.softmax(dim = -2)
        x = self.x(x)
        xa = self.xa(xa)
        x = einsum('b h i j, b h j d -> b h i d', x, va)
        xa = einsum('b h j i, b h j d -> b h i d', xa, v)
        x, xa = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (x, xa))
        out = self.out(x)  
        outxa = self.out(xa) 
           
        return out, outxa, qk

class tgate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        cgate = torch.sum(gates * types.unsqueeze(2), dim=-1)
        return cgate

class residual(nn.Module): 
    def __init__(self, dims: int, head: int, act: str = "silu"):
        super().__init__()

        self.lna = nn.LayerNorm(dims, bias=False)           
        self.atta = attentiona(dims, head)
        self.attb = attentionb(dims, head, max_iter=1)        
        # self.attc = attentiona(dims, head, cross_talk=True)

        self.tgate = tgate(dims, num_types=4)
        self.mlp = nn.Sequential(nn.Linear(dims, dims*4), get_activation(act), nn.Linear(dims*4, dims))

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.atta(x, mask=mask)[0]  
        if xa is not None:
            x = x + self.attb(x, xa, mask=None)
        x = x + self.tgate(x)
        x = x + self.mlp(self.lna(x))
        return x

class processor(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, act: str = "gelu"): 
        super(processor, self).__init__()

        self.ln = nn.LayerNorm(dims)        
        self.token = nn.Embedding(vocab, dims)
        self.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)        
        self.positions = nn.Parameter(torch.empty(ctx, dims), requires_grad=True)

        act_fn = get_activation(act)        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        self.blocka = nn.ModuleList([residual(dims, head, act_fn) for _ in range(layer)])
        self.blockm = nn.ModuleList([residual(dims, head, act_fn) for _ in range(layer // 2)])

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, sequential=False, modal=False, kv_cache=None) -> Tensor:    

        if xa.dim() == 2:
            xa = xa.unsqueeze(0)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token(x.long()) + self.positions[offset : offset + x.shape[-1]])

        xa = self.encoder(xa).permute(0, 2, 1)
        xa = xa + self.audio(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        for block in chain(self.blocka or []):
            xa = block(xa, mask=None)
            x  = block(x, mask=self.mask)
            x  = block(x, xa, mask=None)

        for block in chain(self.blockm or []):
            xm = block(torch.cat([x, xa], dim=1), torch.cat([x, xa], dim=1), mask=None) if modal else None    
            x  = block(xm[:, :x.shape[1]], xm[:, x.shape[1]:], mask=None) if modal else x

        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.ln(x)        
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()
        return x 

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
        
        self.best_loss = float('inf')
        self.factor = nn.Parameter(torch.tensor(2), requires_grad=False)

    def update(self, win_size):
        for name, module in self.processor.named_modules():
            if isinstance(module, (attentionb)):
                module.update_win(win_size)

    def adjust_window(self, loss, ctx):
        self.win_size = ((ctx // self.param.head))
        if loss < self.best_loss:
            win_size = (self.win_size * self.factor) #.clamp(0, ctx - 1)
        else:   
            win_size = (self.win_size // self.factor).clamp(0, self.win_size - 1)
        self.win_size = win_size
        self.best_loss = loss  
        self.update(win_size)
        return win_size
   
    def forward(self, labels=None, input_ids=None, pitch=None, pitch_tokens=None, spectrogram=None):

        x = input_ids
        xa = pitch if pitch is not None else spectrogram         # xb = pitch_tokens     
        logits = self.processor(x, xa)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
        self.adjust_window(loss=loss.item(), ctx=xa.shape[2])
        return {"logits": logits, "loss": loss} 

    def _init_weights(self, module):
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, module in self.named_modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                self.init_counts["LayerNorm"] += 1                
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, residual):
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

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.param.ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, attentiona):
                hooks.append(layer.k.register_forward_hook(save_to_cache))
                hooks.append(layer.v.register_forward_hook(save_to_cache))
        self.processor.apply(install_hooks)
        return cache, hooks
