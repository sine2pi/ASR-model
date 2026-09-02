import os, torch, numpy as np
from torch.nn.functional import scaled_dot_product_attention as SDPA
from typing import Iterable
from torch.nn.utils.parametrizations import weight_norm
from functools import partial
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from optimizerc import FAMScheduler2, MaxFactor
from torch.utils.data import DataLoader    
from datetime import datetime
from essentials import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

_setup_tf32()

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

class AbbyNormal(nn.Module):
    def __init__(n, dims, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0, threshold: float = 0.8):
        super().__init__()
        n.size = size
        n.alpha = alpha
        n.beta = beta
        n.k = k
        n.tx = threshold
        
        n.mode_router = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, 3) 
        )

    def forward(n, x: Tensor, confidence=None) -> Tensor:
        if x.numel() == 0:
            return x

        size = max(3, int(x.size(-1) * 0.05))
        if size % 2 == 0:
            size += 1
        pad_len = size // 2
        
        div = x.mul(x)
        logits = n.mode_router(x)
        mean_val = x.abs().mean(dim=-1, keepdim=True)
        std_val = x.std(dim=-1, keepdim=True)
        cv = std_val / (mean_val + 1e-6)

        decisions = F.gumbel_softmax(logits + cv, tau=1.0, hard=True) 
        avg_d = F.avg_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
        max_d = F.max_pool1d(div.squeeze(0), kernel_size=size, stride=1, padding=pad_len)
  
        div_mode1 = avg_d
        condition = (max_d > 2.0 * avg_d).float()
        div_mode2 = (condition * max_d) + ((1 - condition) * avg_d)
        
        if confidence is None:
            div_mode3 = avg_d
        else:
            conf_mask = (confidence > n.tx).float().unsqueeze(1)
            div_mode3 = (conf_mask * avg_d) + ((1 - conf_mask) * max_d)

        d0 = decisions[..., 0:1] 
        d1 = decisions[..., 1:2] 
        d2 = decisions[..., 2:3] 
        
        div = (d0 * div_mode1) + (d1 * div_mode2) + (d2 * div_mode3)
        denom = div.mul(n.alpha).add(n.k).pow(n.beta)
        out = x / denom
        return out

class ConvLite(nn.Module):
    def __init__(n, dims, kernel_size=15): 
        super().__init__()
        n.point1 = nn.Conv1d(dims, dims * 2, kernel_size=1)
        n.glu = nn.GLU(dim=1) 
        
        n.depth = nn.Conv1d(
            dims, dims, kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, groups=dims
        )
        n.bn = nn.BatchNorm1d(dims) 
        n.swish = nn.SiLU() 
        
        n.point2 = nn.Conv1d(dims, dims, kernel_size=1)
        n.dropout = nn.Dropout(0.1)

    def forward(n, x):
        residual = x
        x = n.point1(x)
        x = n.glu(x)
        x = n.depth(x)
        x = n.bn(x)
        x = n.swish(x)
        x = n.point2(x)
        x = n.dropout(x)
        return residual + x

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, layer, act, n_type, norm=False, enc=False):
        super().__init__()

        n.norm = get_norm(n_type, dims) if norm else nn.Identity()
        n.local_norm = get_norm("localnorm", dims) if norm else nn.Identity()        
        act_fn = get_activation(act)

        n.conv1 = nn.Sequential(
            nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1),
             n.norm
             )
        n.conv2 = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), 
            n.local_norm
            )

        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()

        n.encoder = nn.ModuleList()
        for _ in range(layer):
            n.encoder.append(nn.Sequential(
                act_fn, weight_norm(nn.Conv1d(dims, dims, kernel_size=3, padding=1)),
                LayerNorm(dims), 
                ConvLite(dims, kernel_size=15), 
                act_fn,
                nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn, nn.Dropout(0.1)))

    def _process_feature(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)

        for layer in n.encoder:
            x = layer(x)

        x = x.permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1]).to(device, dtype)
        x = n.norm(x)
        return n.EncoderLayer(x)
               
    def forward(n, x):
        if isinstance(x, TensorDict):
            return x.apply(n._process_feature)
        else:
            return n._process_feature(x)

class rotary(nn.Module):
    def __init__(n, dims, head):
        super().__init__()

        n.head_dim = dims // head
        n.head = head
        n.dims = dims
        n.lin = nn.Linear(dims, n.head_dim // 2, bias=True)

    def gammatone(n, min_freq=200.0, max_freq=8000.0):
        head_dim = n.dims // n.head
        freqs = torch.pow(max_freq / min_freq, torch.linspace(0, 1, head_dim // 2, device=device, dtype=dtype)) * min_freq
        return freqs / 1000

    def wideband(n, max_freq=8000.0):
        head_dim = n.dims // n.head
        mel_max = 2595 * torch.log10(torch.tensor(1 + max_freq / 700, device=device, dtype=dtype))
        mel_scale = torch.pow(10, torch.linspace(0, mel_max, head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 700 * mel_scale / 1000

    def compute_f(n, x=None, mask=None):
        if mask is None:
            scale = gammatone(n.dims, n.head)
            return x.mean(dim=-1) * scale / 1000 if x is not None else 200 * scale / 1000
        else: 
            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype, requires_grad=False) / n.head_dim * torch.log(torch.tensor(x.mean(dim=-1) * THETA if x is not None else THETA, requires_grad=False))

    def forward(n, x=None, xa=None, mask=None): 
        t = torch.arange(x.shape[2], device=device, dtype=dtype).float()
        f = torch.einsum('i,j->ij', t,  n.compute_f(mask=mask))
        m = torch.norm(xa, dim=-1, keepdim=True)
        # m = n.lin(xa)
        # m = torch.sigmoid(n.lin(xa)) ** t
        # m = torch.sigmoid(n.lin(xa)) 

        # this is important, this means something. mashed potatoes.....
        if mask is None:
            f = torch.polar(m, f)
        else: 
            f = torch.polar(m, f)

        x1 = x[..., :f.shape[-1]*2]
        x2 = x[..., f.shape[-1]*2:]
        s = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * f
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(s)
        return torch.cat([x1.type_as(x), x2], dim=-1)
    
class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        n.head = head
        n.head_dim = dims // head
        n.scale = 1.0 / len(features) if features else scale

        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)
    
    def forward(n, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, L, n.head, n.head_dim).transpose(1,2)
        k = n.k(xa).view(B, K, n.head, n.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.head_dim)
        return bias

class attention(nn.Module):
    def __init__(n, dims, head, layer, n_type=None, modal=False): 
        super().__init__()
        n.layer = layer

        n.scale = (dims // head) ** -0.25
        n.modal = modal

        n.q   = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(n_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.c   = nn.Sequential(get_norm(n_type, dims) , nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims))

        n.conv = nn.Conv2d(head, head, 1, bias=False) if modal else nn.Identity()
        n.ln = get_norm(n_type, dims // head)
        n.rot = rotary(dims, head)

    def taylor_softmax(x, order=2):
        ta = 1.0
        for i in range(1, order + 1):
            F_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            ta += x**i / F_i
        return ta / torch.sum(ta, dim=-1, keepdim=True)

    def forward(n, x, xa=None, mask=None, pt=None, window=3, pitch_bias=None): 
        
        b, c, d = x.shape
        k, v = n.kv(aorb(xa, x))
        q = n.q(x)

# potential = ion.mean() + 0.2 * w_metric.mean()

# jump_g = 1.0
# if potential < 0.1 and i < n.layer - 1:
#     action = 1  

        if pitch_bias is not None:
            qk = n.rbf_scores(q * n.scale, k * n.scale, rbf_sigma=1.0, rbf_ratio=0.3)
            pb = pitch_bias(xa) 
            if pb is not None:
                qk = qk + pb[:,:,:q,:q]

            ids = k[:, :, :, 0]
            scale = torch.ones_like(ids)
            fz = torch.clamp(F.softplus(n.fz), n.minz, n.maxz)
            scale[ids.float() == n.pad] = fz
            
            if mask is not None:
                if mask.dim() == 4:
                    mask = mask[0, 0]
                mask = mask[:q, :k] if xa is not None else mask[:q, :q]
                qk = qk + mask * scale.unsqueeze(-2).expand(qk.shape)

            qk = qk * scale.unsqueeze(-2)
            w = F.softmax(qk, dim=-1).to(q.dtype)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if pt is not None:
            c = n.c(pt)
            b, h, c, d = q.shape 
            t = torch.zeros(b, h, c, c, device=device, requires_grad=False)

            for i in range(c):
                for j in range(c):
                    start = max(0, min(i, j) - window)
                    end = min(c, max(i, j) + window)
                    
                    for k in range(start, end): 
                        score = (q[:, :, i, :] * k[:, :, j, :] * c[:, :, k, :]).sum(dim=-1)
                        t[:, :, i, j] += score

            q = q * n.scale + t
            k = k * n.scale + t

        else:
            q = q * n.scale 
            k = k * n.scale

        q, k = n.rot(q, xa=x if pt is None else pt, mask=mask), n.rot(k, xa=xa if xa is not None else x, mask=mask)  
        a = SDPA(n.ln(q), n.ln(k), v, is_causal=have(mask))

        if n.modal and xa is not None:
            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb = n.q(x), n.q(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)
            b = SDPA(n.ln(qa), n.ln(kb), vb, is_causal=have(mask))
            c = SDPA(n.ln(qb), n.ln(ka), va, is_causal=have(mask))
            return n.out(a), n.out(n.conv(b)), n.out(n.conv(c))
        else:
            return n.out(a)

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

class v_gate(nn.Module):
    def __init__(n, dims, mem=64, thresh=0.5):
        super().__init__()
        n.mkey = nn.Parameter(torch.randn(mem, dims))
        n.mval = nn.Parameter(torch.randn(mem, 1))
        n.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        
        n.tx = nn.Parameter(torch.tensor(thresh, dtype=dtype), requires_grad=False)
        n.concat = nn.Linear(2, 1, device=device, dtype=dtype)

    def forward(n, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x_val = n.concat(torch.cat((torch.matmul(key, n.mval), n.mlp(x)), dim=-1))
        
        smask = apply_ste(x_val, n.tx)
        return smask, x_val

    def update_threshold(n, loss, cema, lr=0.01):
        if loss > cema:
            n.tx.sub_(lr)
        else:
            n.tx.add_(lr)
        n.tx.data = torch.clamp(n.tx.data, 0.05, 0.95)

class r_node(nn.Module):
    def __init__(n, dims, exp=2):
        super().__init__()
        n.dims = dims
        n.exp = exp
        n.par = nn.ModuleList([nn.Linear(dims, dims) for _ in range(exp)])
        n.net = nn.Linear(dims, dims)
        n.relu = SnakeActivation(alpha=1.0) if exp == 3 else nn.ReLU()
        # n.relu = nn.ReLU() 

    def forward(n, x):
        feat = torch.stack([path(x) for path in n.par])
        wts = torch.softmax(n.net(x), dim=-1)
        wtd =  torch.sum(wts * feat.unsqueeze(2), dim=-1)
        return n.relu(wtd)

class SnakeActivation(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)

# class r_node2(nn.Module):
#     def __init__(n, dims, exp=2):
#         super().__init__()
#         n.dims = dims
#         n.exp = exp
#         n.par = nn.ModuleList([nn.Linear(dims, dims) for _ in range(exp)])
#         n.net = nn.Linear(dims, dims)
        
#         n.relu_fn = nn.ReLU()
#         n.snake_fn = SnakeActivation(alpha=1.0)

#     def forward(n, x, condition_metric=None):

#         feat = torch.stack([path(x) for path in n.par], dim=0) # [exp, B, L, D]
#         wts = torch.softmax(n.net(x), dim=-1) # [B, L, exp]
#         feat_perm = feat.permute(1, 2, 0, 3) 
#         wtd = torch.sum(wts.unsqueeze(-1) * feat_perm, dim=-2) # [B, L, D]

#         if condition_metric is None:
#             token_variance = x.std(dim=-1).mean()
#             use_snake = (token_variance > 1.5).float()

#         else:
#             use_snake = (condition_metric < 0.5).float()

#         relu_out = n.relu_fn(wtd)
#         snake_out = n.snake_fn(wtd)
#         return use_snake * snake_out + (1.0 - use_snake) * relu_out

# if layer['ranvier'] is not None:
#     out = layer['ranvier'](apx, condition_metric=potential)
# else:
#     out = apx

# class SnakeActivation(nn.Module):
#     def __init__(self, alpha=1.0):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

#     def forward(self, x):
#         return x + (1.0 / self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)

# class r_node3(nn.Module):
#     def __init__(n, dims, exp=2):
#         super().__init__()
#         n.dims = dims
#         n.exp = exp
#         n.par = nn.ModuleList([nn.Linear(dims, dims) for _ in range(exp)])
#         n.net = nn.Linear(dims, dims)
        
#         n.relu_fn = nn.ReLU()
#         n.snake_fn = SnakeActivation(alpha=1.0)
        
#         n.metric_to_gate = nn.Linear(1, 1)
#         n.register_buffer("snake_selection_weight", torch.zeros(1), persistent=False)

#     def forward(n, x, condition_metric=None, tracked_dict=None):
#         # x shape: [B, L, D]
#         feat = torch.stack([path(x) for path in n.par], dim=0) # [exp, B, L, D]
#         wts = torch.softmax(n.net(x), dim=-1) # [B, L, exp]
        
#         feat_perm = feat.permute(1, 2, 0, 3) 
#         wtd = torch.sum(wts.unsqueeze(-1) * feat_perm, dim=-2) # [B, L, D]

#         if condition_metric is None:
#             mean_val = x.abs().mean(dim=-1, keepdim=True)
#             std_val = x.std(dim=-1, keepdim=True)
#             metric_tensor = std_val / (mean_val + 1e-6) # [B, L, 1]
#         else:
#             if condition_metric.dim() == 0:
#                 metric_tensor = condition_metric.view(1, 1, 1).expand(x.size(0), x.size(1), 1)
#             elif condition_metric.dim() == 1:
#                 metric_tensor = condition_metric.view(-1, 1, 1).expand(-1, x.size(1), 1)
#             else:
#                 metric_tensor = condition_metric
#         gate_logits = n.metric_to_gate(metric_tensor)
#         snake_prob = torch.sigmoid(gate_logits) # [B, L, 1]

#         with torch.no_grad():
#             avg_prob = snake_prob.mean().item()
#             n.snake_selection_weight.copy_(torch.tensor(avg_prob, device=x.device))
            
#             if isinstance(tracked_dict, dict):
#                 if 'snake_usage_history' not in tracked_dict:
#                     tracked_dict['snake_usage_history'] = []
#                 tracked_dict['snake_usage_history'].append(avg_prob)

#         relu_out = n.relu_fn(wtd)
#         snake_out = n.snake_fn(wtd)
        
#         return snake_prob * snake_out + (1.0 - snake_prob) * relu_out

    # def forward(n, x, tracked_dict=None):

    #     while i < n.layer:
    #         layer = n.layers[i]

    #         if layer['ranvier'] is not None:
    #     
    #             out = layer['ranvier'](apx, condition_metric=potential, tracked_dict=tracked_dict)
    #         else:
    #             out = apx

    # step_metrics = {}

    # output = model(text_ids=text_batch, spectrogram=spec_batch, tracked_dict=step_metrics)
    
    # if 'snake_usage_history' in step_metrics:
    #     mean_snake_ratio = np.mean(step_metrics['snake_usage_history'])
    #     print(f"Step Activation Profiles -> Snake Weight Ratio: {mean_snake_ratio:.4f} | ReLU Ratio: {1.0 - mean_snake_ratio:.4f}")

class MPNet(nn.Module):
    def __init__(n, dims, jump=2):
        super().__init__()
        n.net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.SiLU(),
            nn.Linear(128, jump + 1)
        )
        
    def forward(n, pooled):
        return F.softmax(n.net(pooled), dim=-1)

class MSheath(nn.Module):
    def __init__(n, dims, head, layer, mini_hc, rate):
        super().__init__()
        n.layer = layer
        n.dims = dims
        n.l_jump = True  
        n.jstat = {0: 0, 1: 0, 2: 0} 
        
        # n.shared_head = AdaptiveSpan(dims, head, max_dist=rate, sharpen=True, temp_scale=0.01)
        n.mem_w = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        n.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        
        n.jump_s = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)
        
        n.layers = nn.ModuleList()
        for i in range(layer):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'v_gate': v_gate(dims, mem=64, thresh=0.3),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
            }

            if mini_hc:
                layer_dict['ranvier'] = r_node(dims, exp=rate)
            else:
                layer_dict['ranvier'] = None

            n.layers.append(nn.ModuleDict(layer_dict))

        n.pnet = MPNet(dims, jump=2)
        # n.oneshot = OneShot(dims, head)

        n.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        n.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4), 
            nn.SiLU(), 
            nn.Linear(dims * 4, dims)
        )

        n.mlp_ln = nn.LayerNorm(dims)
        # n.dendrites = AdaptiveSpan(dims, head, max_dist=1, sharpen=True, temp_scale=0.01)
    
    def forward(n, x): 
        
        batch, ctx = x.shape[:2]
        orig_x = x
        
        mem_w = n.mem_w.expand(batch, -1, -1)
        pooled = x.mean(dim=1)
        policy = n.pnet(pooled)
        
        history = []
        i = 0

        while i < n.layer:
      
            layer = n.layers[i]
            
            ion, _ = layer['v_gate'](x)
            mlayer = ion.expand(-1, ctx, n.dims)
            
            px = layer['ln'](x)  

            if layer['adapter'] is not None:
                apx = layer['adapter'](px)
            else:
                apx = px

            if layer['ranvier'] is not None:
                out = layer['ranvier'](apx)
            else:
                out = apx
                
            g_val = layer['gate'](px)
            x = x + g_val * (out * mlayer)

            mem = x.mean(dim=1, keepdim=True)
            mem_v = n.mem_gate(mem)
            mem_w = mem_v * mem_w + (1 - mem_v) * mem
            
            potential = ion.mean()
            jump_g = 1.0
            
            if potential < 0.1 and i < n.layer - 1:
                action = 1
                
            elif i < n.layer - 1:
                if n.l_jump:

                    jump = F.gumbel_softmax(policy, tau=1.0, hard=True)
                    action = jump.argmax(dim=-1).item()
                    jump_g = jump[0, action] 
                else:    
                    action = torch.multinomial(policy, 1).squeeze(-1).item()
            else:
                action = 0
                
            if action in n.jstat:
                n.jstat[action] += batch
            else:
                n.jstat[action] = batch
                
            if action > 0:
                jdist = action
                i_next = min(i + jdist + 1, n.layer)
                jump_w = n.jump_s[min(jdist-1, 2)]               
                jump_i = jump_w * orig_x + (1-jump_w) * mem_w.expand(-1, ctx, -1)
                x = x + (jump_i * jump_g) 
                
                i = i_next
                history.append({'layer': i, 'status': 'jumped_to'})
            else:
                x = x * jump_g
                i += 1
                history.append({'layer': i, 'status': 'processed'})
        
        # wint = warp_stats.mean() 
        # eff = ion.mean() + 0.5 * wint

        # jump_g = 1.0
        # if eff < 0.1 and i < n.layer - 1:
        #     action = 1
        # elif i < n.layer - 1:
        #     if n.l_jump:
        #         adjusted_policy = policy + warp_stats.view_as(policy)
        #         jump = F.gumbel_softmax(adjusted_policy, tau=1.0, hard=True)
        #         action = jump.argmax(dim=-1).item()
        #         jump_g = jump[0, action]
        #     else:
        #         action = torch.multinomial(policy, 1).squeeze(-1).item()
        # else:
        #     action = 0

        # x = n.dendrites(x)
        gate = n.mlp_gate(x)
        output = n.mlp(n.mlp_ln(x))
        x = x + gate * output
        jmp = {'jump_history': history}
        return x, jmp

class gate(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        n.features = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.top = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x, num=2):
        types, indices = torch.topk(n.top(x), num, dim=-1)
        type = torch.zeros_like(n.features(x))
        type.scatter_(-1, indices, torch.nn.functional.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class tgate(nn.Module):
    def __init__(n, dims, num_types=2):
        super().__init__()

        n.ga = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        n.cs = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))

    def forward(n, x):
        types = n.cs(x)
        ga = torch.stack([g(x) for g in n.ga], dim=-1)
        return  torch.sum(ga * types.unsqueeze(2), dim=-1)

class router(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()
        n.num_types = num_types
        n.top = nn.Linear(dims * num_types, num_types)
        n.soft = nn.Sequential(nn.Linear(dims * num_types, num_types), nn.Softmax(dim=-1))
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def weights(n, router_top, router_input, num=2):
        types, indices = torch.topk(router_top, num, dim=-1)
        type = torch.zeros_like(router_top)
        type.scatter_(-1, indices, F.softmax(types, dim=-1))
        soft_selection = n.soft(router_input)
        alpha = torch.sigmoid(n.alpha)
        return alpha * type + (1 - alpha) * soft_selection

    def forward(n, *modalities):
        stack = torch.stack(modalities, dim=-1)
        input = stack.view(stack.shape[0], stack.shape[1], -1)
        weights = n.weights(n.top(input), input)
        return torch.sum(stack * weights.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, num_types=3):
        super().__init__()

        n.layer = layer - 1 
        n.ln = get_norm(n_type=n_type, dims=dims)

        n.act_fn = get_activation(act)
        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        
        n.attn = attention(dims, head, layer, n_type=n_type)
        n.router = router(dims, num_types=num_types)
        n.jump = MSheath(dims, head, layer, mini_hc=False, rate=num_types)

        n.mlp = nn.Sequential(n.ln, tgate(dims, num_types=num_types), 
                              nn.Linear(dims, dims*num_types), get_activation(act), nn.Linear(dims*num_types, dims), n.ln)

    def forward(n, x, xa=None, mask=None, pt=None):
        x, jmp  = n.jump(n.ln(x))
        x = n.router(*[x for _ in range(n.layer)]) + n.attn(n.ln(x), mask=mask, pt=pt)
        if xa is not None:
            xa = xa + n.audio(xa.shape[1], xa.shape[-1]).to(device, dtype)
            xa, jmp = n.jump(n.ln(xa))
            x = x + n.attn(n.ln(x), xa=n.router(*[xa for _ in range(n.layer)]), pt=pt)

        print(jmp['jump_history']) # this will be noisy
        return x + n.mlp(x).to(device, dtype)

class processor(nn.Module):
    def __init__(n, tokens, mels, dims, head, layer, act, n_type, ctx=2048): 
        super().__init__()

        n.dims = dims
        n.ln = get_norm(n_type, dims)

        n.token = nn.Embedding(tokens, dims)
        n.pitch_tokens = nn.Embedding(1024, dims) 
        n.position = nn.Parameter(torch.ones(ctx, dims), requires_grad=True)
        n.blend = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        n.block: Iterable[residual] = nn.ModuleList(
            [residual(dims, head, layer, act, n_type) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, seq=False) -> Tensor:
        blend = torch.sigmoid(n.blend)
        mask = n.mask[:x.shape[1], :x.shape[1]]

        x1 = n.token(x)    

        if xa['pt'] is not None: # skipping pt for now 
            pt = n.quantize_pitch(pt=xa['pt'])
            x2 = n.pitch_tokens(pt)
            x1 = x1 + x2 
        else:
            pt = None 

        x = (x1 + n.position[:x.shape[-1]]).to(device, dtype)
        # x = (x1 + n.position[:x.shape[1]]).to(device, dtype)



        for i in n.block:
            a = i(x, mask=mask, pt=pt)
            b = i(a, xa=i(xa['a']), pt=pt)
            c = i(b, xa=i(xa['b']), pt=pt)
            d = i(c, xa=i(xa['c']), pt=pt)

            # for j in [(xa['a']), (xa['b']), (xa['c'])]: e = i(x, xa=i(j, pt=pt))
            # e = torch.mean(torch.stack([xa['a'], xa['b'], xa['c']]), dim=0)

            e = a + b + c
            f = torch.cat([d, e], dim=1)

#         g = i(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:], pt=pt)
        
#         if seq:
#             x = g
#         else:
#             x = blend * d + (1 - blend) * g if g is not None else a

            g = i(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:])
        
        x = g if seq else blend * (d) + (1 - blend) * g if g is not None else a
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

        n.enc = AudioEncoder(param.mels, param.dims, param.head, param.layer, param.act, param.n_type, norm=False, enc=False)

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
            'pt': pitch_tokens,
            }, batch_size=fx.shape[0])

        x = text_ids
        xa = n.enc(no_none(xa))
        output = n.processor(x, xa, seq=False)

        loss = None
        if labels is not None: 
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": output, "loss": loss}

def main():

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    metadata_file = "H:/DEV/datasets/lb1/metadata.csv"
    data_dir = "H:/DEV/datasets/lb1"

    log_dir = os.path.join('./logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("H:/DEV/sam3_motion/sine2pi/ASR-model/tokenizer.json") 
    
    extract_args = {

        "spectrogram": True,
        "pitch": True,
        "waveform": True,
        "pitch_tokens": False,
        "harmonics": False,
        "aperiodics": False,
        "hop_length": 160,
        "sample_rate": 16000,
        "mels": 128
    }

    param = Dimensions(tokens=40000, mels=128, dims=512, head=4, layer=4, act="gelu", n_type="AbbyNormal")

    dataset = prepare_datasets(metadata_file, data_dir, tokenizer, extract_args=extract_args)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    model = Model(param).to('cuda')

    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)
    Collator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=1, 
        collate_fn=Collator, 
        num_workers=0,
    )

    eval_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=1, 
        collate_fn=Collator, 
        num_workers=0,
    )

    main_params = []
    jump_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'jump' in name or 'pnet' in name or 'micro_filter' in name:
            jump_params.append(p)
        else:
            main_params.append(p)

    optimizer = MaxFactor([
        {'params': main_params, 'bias': 1.0}, 
        {'params': jump_params, 'bias': 2.0}  
    ], lr=2.5e-3, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=1e-2, gamma=0.99, max=False, bias=1, 
                 min_lr=1e-9, clip=False, cap=0.0)

    # optimizer = MaxFactorA(model.named_parameters(), lr=2.5e-3, b_decay=-0.8, eps=(1e-8, 1e-8), d=1.0, decay=1e-2, gamma=0.99, max=False, clip=False, cap=0.0)

    scheduler = FAMScheduler2(optimizer, warmup_steps=10, total_steps=100, 
                 decay_start=None, warmup_start=1e-6, eta_min=1e-6, last_epoch=-1) 

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    train_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_dataloader,
        eval_loader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metric_fn=metrics_fn,
        max_steps=100,
        device="cuda",
        acc_steps=1,
        clear_cache=False,
        log_interval=10,
        eval_interval=10,
        save_interval=0,
        warmup_interval=10,
        checkpoint_dir=log_dir,
        log_dir=log_dir,
        generate=False,
        clip_grad_norm=0.0,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
