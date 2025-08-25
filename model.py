
import warnings
import logging
import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from typing import Optional, Iterable
import numpy as np
from dataclasses import dataclass
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def have(a):
    return a is not None

def AorB(a, b):
    return a if have(a) else b

def valid(default_value, *items):
    for item in items:
        if item is not None:
            return item
    return default_value

def dict_to(d, device, dtype=dtype):
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}  

class InstanceRMS(nn.Module):
    def __init__(n, num_features, eps=1e-6):
        super().__init__()
        n.instance_norm = nn.InstanceNorm1d(
            num_features,
            eps=eps,
            affine=False,
            track_running_stats=False
        )
        n.gamma = nn.Parameter(torch.ones(num_features))

    def forward(n, x):
        normalized_x = n.instance_norm(x)
        gamma_reshaped = n.gamma.view(1, -1, 1)
        output = normalized_x * gamma_reshaped
        return output

class AdaLayerNorm(nn.Module):
    def __init__(n, num_embeddings: int, eps: float = 1e-6):
        super().__init__()
        embedding_dim = num_embeddings
        n.eps = eps
        n.dim = embedding_dim
        n.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        n.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(n.scale.weight)
        torch.nn.init.zeros_(n.shift.weight)

    def forward(n, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = n.scale(cond_embedding_id)
        shift = n.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (n.dim,), eps=n.eps)
        x = x * scale + shift
        return x   

def calculate_attention(q, k, v, mask=None, temp=1.0):
    b, h, c, d = q.shape
    scaled_q = q
    if temp != 1.0 and temp > 0:
        scaled_q = q * (1.0 / temp)**.5
    out = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and c > 1)        
    return out

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

def get_norm(norm_type: str, num_features: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if norm_type in ["batchnorm", "instancenorm"] and num_features is None:
        raise ValueError(f"'{norm_type}' requires 'num_features'.")
    if norm_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{norm_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=num_features, bias=False),
        "AdaLayerNorm": lambda: AdaLayerNorm(num_embeddings=num_features),
        "InstanceRMS": lambda: InstanceRMS(num_features=num_features),        
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=num_features),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=num_features),
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=num_features),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=num_features),
        }
   
    norm_func = norm_map.get(norm_type.lower())
    if norm_func:
        return norm_func()
    else:
        print(f"Warning: Norm type '{norm_type}' not found. Returning LayerNorm.")
        return nn.LayerNorm(num_features) 

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(n, channels):
        super().__init__()
        n.alpha = nn.Parameter(torch.ones(1, channels, 1))

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
        "PeriodicReLU": PeriodicReLU,
        # "snake": Snake1d,
    }
    return act_map.get(act, nn.GELU())

def sinusoids(ctx, dims, theta=10000):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding

@dataclass
class Dimensions:
    tokens: int
    mels: int
    ctx: int
    dims: int
    head: int
    layer: int
    act: str
    norm_type: str

class rotary(nn.Module):
    def __init__(n, dims, head):
        super(rotary, n).__init__()
        n.dims = dims
        n.head = head
        n.head_dim = dims // head

        n.theta = nn.Parameter((torch.tensor(10000, device=device, dtype=dtype)), requires_grad=True)  
        n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(n):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(n, x) -> Tensor:
        freqs = (n.theta / 220.0) * n.freqs_base 

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
        weighted_combined = sum(w * f for w, f in zip(weights.unbind(dim=-1), features))
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
        for i in range(layer):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
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
        i = 0
        while i < n.layer:
            layer = n.layers[i]
            
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
            
            if i < n.layer - 1:
                action = torch.multinomial(policy, 1).squeeze(1).item()
            else:
                action = 0
            distance = 0
            if action == 1: distance = 1
            if action == 2: distance = 2
            if distance > 0:
                i_next = min(i + distance, n.layer - 1)
                jump = n.jump_weights[min(distance-1, 2)]               
                x = x + jump * ox + (1-jump) * work_mem.expand(-1, ctx, -1)
                i = i_next
                history.append(i)
            else:
                i += 1
        
        x3 = n.mlp_gate(x)
        output = n.mlp(n.mlp_ln(x))
        x = x + x3 * output
        n.logs = {'jumps': history}
        return x
        
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

        q, k = map(n.rotary.rotate_queries_or_keys, (q, k))

        iteration = 0
        temp = n.temp
        prev_out = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = n.threshold
        curq = q
        
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

            attn_out = n._focus(x=qwin, xa=kwin, mask=win_mask, win_size=win_size)
            out[:, qstart:qend, :] = attn_out
        return out

    def forward(n, x, xa = None, mask = None):
        x = n._slide_win_local(x, mask=None)
        xa = n._slide_win_local(xa, mask=None)
        out = n._focus(x, xa, mask=None)
        return n.out(out)
        
def compute_freqs_base(dim):
    mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 200 * mel_scale / 1000 

class attention(nn.Module):
    def __init__(n, dims: int, head: int, layer: int, norm_type):
        super().__init__()

        n.q   = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        n.register_buffer('freqs_base', compute_freqs_base(dims // head), persistent=False)
        n.rotary = RotaryEmbedding(dims // head, custom_freqs = (36000 / 220.0) * n.freqs_base)  
        n.ln = get_norm(norm_type, dims // head)

    def forward(n, x, xa = None, mask = None):
        b, c, d = x.shape
        q = n.q(x)
        k, v = n.kv(AorB(xa, x))

        if xa is None:
            q = n.rotary.rotate_queries_or_keys(q)
            k = n.rotary.rotate_queries_or_keys(k)

        wv = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=mask is not None and c > 1)
        out = n.out(wv)
        return out

class AttentionA(nn.Module):
    def __init__(n, dims: int, head: int):
        super().__init__()
        n.head = head
        n.dims = dims
        n.head_dim = dims // head
        
        n.ln = nn.LayerNorm(dims)
        n.q = nn.Linear(dims, dims, bias=False)
        n.kv = nn.Linear(dims, dims * 2, bias=False)
        n.out = nn.Linear(dims, dims, bias=False)
        
        n.x_conv = nn.Conv2d(head, head, 1, bias=False)
        n.xa_conv = nn.Conv2d(head, head, 1, bias=False)

        n.register_buffer('freqs_base', compute_freqs_base(dims // head), persistent=False)
        n.rotary = RotaryEmbedding(dims // head, custom_freqs = (36000 / 220.0) * n.freqs_base)  

    def forward(n, x, xa=None, mask=None):

        q = n.q(n.ln(x))
        k, v = n.kv(n.ln(x)).chunk(2, dim=-1)
        ka, va = n.kv(n.ln(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=n.head), (q, k, v))
        ka, va = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=n.head), (ka, va))

        if xa is None:
            q, k = map(n.rotary.rotate_queries_or_keys, (q, k))
       
        attn_weights_x = torch.einsum('b h i d, b h j d -> b h i j', q, ka)
        attn_probs_x = F.softmax(attn_weights_x, dim=-1)
        x_updated = torch.einsum('b h i j, b h j d -> b h i d', attn_probs_x, va)
        
        attn_weights_xa = torch.einsum('b h j d, b h i d -> b h j i', ka, q)
        attn_probs_xa = F.softmax(attn_weights_xa, dim=-1)
        xa_updated = torch.einsum('b h j i, b h i d -> b h i d', attn_probs_xa, v)

        x_updated = rearrange(x_updated, 'b h n d -> b n (h d)')
        xa_updated = rearrange(xa_updated, 'b h n d -> b n (h d)')
        return n.out(x_updated), n.out(xa_updated)

class attentionB(nn.Module):
    def __init__(n, dims: int, head: int, layer: int, norm_type):
        super().__init__()

        n.q   = nn.Sequential(get_norm(norm_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h n d -> b n (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        
        n.register_buffer('freqs_base', compute_freqs_base(dims // head), persistent=False)
        n.rotary = RotaryEmbedding(dims // head, custom_freqs = (36000 / 220.0) * n.freqs_base)  
        n.ln = get_norm(norm_type, dims // head)
        
    def forward(n, x, xa=None, mask=None):
        
        q = n.q(x)
        k, v = n.kv(AorB(xa, x))
        _, _, c, d = q.shape 
        scale = d ** -0.5

        q, k = map(n.rotary.rotate_queries_or_keys, (q, k))
        qk = einsum('b h k d, b h q d -> b h k q', n.ln(q), n.ln(k)) * scale

        if have(mask):
            qk = qk + mask[:c, :c]

        qk = F.softmax(qk, dim=-1, dtype=dtype)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        return n.out(wv)
        
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

        x_fused, xa_fused = n.cross_modal(x, xa, mask=mask)
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
        memory = torch.cat([x, xa], dim=1)
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
        
class residual(nn.Module): 
    def __init__(n, dims, head, layer, act, norm_type):
        super().__init__()
        n.ln = get_norm(norm_type, dims)
        n.attn = attention(dims, head, layer, norm_type)
        n.mlp = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims*4), get_activation(act), nn.Linear(dims*4, dims))

    def forward(n, x, xa=None, mask=None):
        x = x + n.attn(n.ln(x), mask=mask) 
        if xa is not None:
            x = x + n.attn(n.ln(x), xa)
        return x + n.mlp(x)
  
class HyperBlock(nn.Module):
    def __init__(n, dims, head, layer, act, norm_type, expand=1):
        super().__init__()
        n.dims = dims
        n.expand = expand
        n.main = residual(dims, head, layer, act, norm_type)
        n.pathways = nn.ModuleList([residual(dims, head, layer, act, norm_type) for _ in range(expand)])
        n.net = nn.Linear(dims, expand)
        n.act = get_activation(act)
        
    def forward(n, x, xa=None, mask=None):
        out = n.main(x, xa=xa, mask=mask)
        eo = [pathway(x, xa=xa, mask=mask) for pathway in n.pathways]
        wts = torch.softmax(n.net(x), dim=-1)
        wo = sum(w * f for w, f in zip(wts.unbind(dim=-1), eo))
        out = n.act(out + wo)
        return out
        
class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, norm_type, feature=None, norm=False):
        super().__init__()
        
        n.norm = norm
        n.act_fn = get_activation(act)

        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True)

        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        n.encoder = nn.Sequential(n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), n.act_fn, 
        nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   
        
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
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device=device, dtype=dtype)

        if n.norm:
            x = n.ln(x)

        return n.EncoderLayer(x)

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

class processor(nn.Module):
    def __init__(n, tokens, mels, ctx, dims, head, layer, act, norm_type): 
        super().__init__()

        n.token = nn.Embedding(tokens, dims) 
        n.positions = nn.Parameter(torch.empty(ctx, dims))
        n.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)   
        n.ln = get_norm(norm_type, dims)
        
        use_hyper_connections = False
        if use_hyper_connections:
            n.block: Iterable[HyperBlock] = nn.ModuleList([HyperBlock(dims=dims, head=head, layer=layer, act=act, norm_type=norm_type, rate=2) for _ in range(layer)])
        else:        
            n.block: Iterable[residual] = nn.ModuleList([residual(dims=dims, head=head, layer=layer, act=act, norm_type=norm_type) for _ in range(layer)])
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa, kv_cache=None) -> Tensor:    

        mask = n.mask[:x.shape[1], :x.shape[1]]
        xa = xa + n.audio(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (n.token(x)+ n.positions[offset : offset + x.shape[-1]])

        for i in n.block:
            xa = i(xa, xa=None, mask=None)
            x  = i(x, xa=None, mask=mask)
            x  = i(x, xa=xa, mask=None)

        return (n.ln(x) @ torch.transpose(n.token.weight.to(dtype), 0, 1)).float()

class Model(nn.Module):
    def __init__(n, param: Dimensions):
        super().__init__()
        n.param = param
        n.processor = processor(
            tokens=param.tokens,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            layer=param.layer,
            act=param.act,
            norm_type=param.norm_type,
            )       
        
        n.FEncode = AudioEncoder(param.mels, param.dims, param.head, act=param.act, norm_type=param.norm_type, norm=False)

        n.layer_count = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer_count += 1        

    def forward(n, labels=None, input_ids=None, pitch=None, pitch_tokens=None, spectrogram=None, waveform=None):

        spectrogram = n.FEncode(spectrogram) if spectrogram is not None else None
        pitch = n.FEncode(pitch) if pitch is not None else None       

        logits = n.processor(input_ids, AorB(pitch, spectrogram))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": logits, "loss": loss} 

    def _init_weights(n, m):
        n.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0, "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, m in n.named_modules():
            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
                n.init_counts["RMSNorm"] += 1
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                n.init_counts["LayerNorm"] += 1                
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                n.init_counts["Linear"] += 1

    def init_weights(n):
        print("Initializing model weights...")
        n.apply(n._init_weights)
        print("Initialization summary:")
        for module_type, count in n.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")
