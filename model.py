import os
import warnings
import logging
from itertools import chain
import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from typing import Dict, Iterable, Optional, Tuple, Union, Tuple, List
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from einops.layers.torch import Rearrange
from echoutils import *
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

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
        n.reset_parameters()

    def forward(n, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(n.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x = n.concat(torch.cat((torch.matmul(key, n.mval),  n.mlp(x)), dim=-1))
       
        threshold = apply_ste(x, n.threshold)
        return threshold, x

    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class MiniConnection(nn.Module):
    def __init__(n, dims, expand=2):
        super().__init__()
        n.dims = dims
        n.expand = expand
        n.parallel = nn.ModuleList([nn.Linear(dims, dims) for _ in range(expand)])
        n.network = nn.Linear(dims, expand)
        n.relu = nn.ReLU()
        n.reset_parameters()

    def forward(n, input_features):
        features = [pathway(input_features) for pathway in n.parallel]
        weights = torch.softmax(n.network(input_features), dim=-1)
        weighted_combined = sum(w * f for w, f in zip(weights.unbind(dim=-1), features))
        return n.relu(weighted_combined)
        
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

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
        n.reset_parameters()

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
        
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

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
        n.reset_parameters()

        n.max_iter = max_iter
        n.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
        n.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        

        n.local = LocalOut(dims, head)   
        n.reset_parameters()

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
        
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

def compute_freqs_base(dim):
    mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), dim // 2, device=device, dtype=dtype) / 2595) - 1
    return 200 * mel_scale / 1000 

class attention(nn.Module):
    def __init__(n, dims: int, head: int, layer: int, norm_type):
        super().__init__()

        n.q   = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims), nn.Dropout(0.01))
        n.register_buffer('freqs_base', compute_freqs_base(dims // head), persistent=False)
        n.rotary = RotaryEmbedding(dims // head, custom_freqs = (36000 / 220.0) * n.freqs_base)  
        n.ln = nn.LayerNorm(dims // head)
        n.reset_parameters()

    def forward(n, x, xa = None, mask = None):
        q = n.q(x)
        k, v = n.kv(AorB(xa, x))

        if xa is None:
            q = n.rotary.rotate_queries_or_keys(q)
            k = n.rotary.rotate_queries_or_keys(k)

        wv = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=mask is not None and q.shape[2] > 1)
        out = n.out(wv)
        return out
       
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class residual(nn.Module): 
    def __init__(n, dims, head, layer, act, norm_type):
        super().__init__()
        n.ln = get_norm(norm_type, dims)
        n.attn = attention(dims, head, layer, norm_type)
        n.mlp = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims*4), get_activation(act), nn.Linear(dims*4, dims))
        n.gate = tgate(dims, num_types=2)
        n.reset_parameters()

    def forward(n, x, xa=None, mask=None):
        x = x + n.attn(n.ln(x), mask=mask) 
        if xa is not None:
            x = x + n.attn(n.ln(x), xa)
        return x + n.mlp(x)
        
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class HyperBlock(nn.Module):
    def __init__(n, dims, head, layer, act, norm_type, expand=1):
        super().__init__()
        n.dims = dims
        n.expand = expand
        n.main = residual(dims, head, layer, act)
        n.pathways = nn.ModuleList([residual(dims, head, layer, act, norm_type) for _ in range(expand)])
        n.net = nn.Linear(dims, expand)
        n.act = get_activation(act)
        n.reset_parameters()

    def forward(n, x, xa=None, mask=None):
        out = n.main(x, xa=xa, mask=mask)
        eo = [pathway(x, xa=xa, mask=mask) for pathway in n.pathways]
        wts = torch.softmax(n.net(x), dim=-1)
        wo = sum(w * f for w, f in zip(wts.unbind(dim=-1), eo))
        out = n.act(out + wo)
        return out
        
    def reset_parameters(n) -> None:
        for m in n.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)        

class FEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, norm_type):
        super().__init__()
        
        act_fn = get_activation(act)
        n.encoder = nn.Sequential(
           nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
           nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)
        
    def forward(n, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)        
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device=device, dtype=dtype)
        return x

class processor(nn.Module):
    def __init__(n, tokens, mels, ctx, dims, head, layer, act, norm_type): 
        super().__init__()

        n.token = nn.Embedding(tokens, dims) 
        n.positions = nn.Parameter(torch.empty(ctx, dims))
        n.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)   
        n.ln = get_norm(norm_type, dims)
        
        use_hyper_connections = False
        if use_hyper_connections:
            n.block = nn.ModuleList([HyperBlock(dims=dims, head=head, layer=layer, act=act, norm_type=norm_type, rate=2) for _ in range(layer)])
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
        
        n.FEncode = FEncoder(param.mels, param.dims, param.head, act=param.act, norm_type=param.norm_type)
        n.PEncode = PEncoder(1, param.dims, param.head, act=param.act)
        n.WEncode = WEncoder(1, param. dims, param.head, act=param.act)

        n.layer_count = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer_count += 1        

    def forward(n, labels=None, input_ids=None, pitch=None, pitch_tokens=None, spectrogram=None, waveform=None):
        spectrogram = n.FEncode(spectrogram) if spectrogram is not None else None
        waveform = n.WEncode(waveform) if waveform is not None else None
        pitch = n.FEncode(pitch) if pitch is not None else None        
        logits = n.processor(input_ids, AorB(pitch, spectrogram))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
        return {"logits": logits, "loss": loss} 

    def _init_weights(n, module):
        n.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0
            }
        for name, module in n.named_modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
                n.init_counts["RMSNorm"] += 1
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                n.init_counts["LayerNorm"] += 1                
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                n.init_counts["Linear"] += 1
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                n.init_counts["Conv1d"] += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                n.init_counts["Conv2d"] += 1
            elif isinstance(module, residual):
                n.init_counts["Residual"] += 1
            elif isinstance(module, processor):
                n.init_counts["processor"] += 1

    def init_weights(n):
        print("Initializing model weights...")
        n.apply(n._init_weights)
        print("Initialization summary:")
        for module_type, count in n.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

    def install_kv_cache_hooks(n, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > n.param.ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, attention):
                hooks.append(layer.k.register_forward_hook(save_to_cache))
                hooks.append(layer.v.register_forward_hook(save_to_cache))
        n.processor.apply(install_hooks)
        return cache, hooks

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=True, load_saved=False, save_dataset=True, cache_dir='E:/hf', extract_args=None, max_ctx=2048):

    PATH = 'E:/hf'
    os.environ['HF_HOME'] = 'E:/hf'
    os.environ['HF_DATASETS_CACHE'] = 'E:/hf'

    if load_saved:
        if cache_dir is None:
            cache_dir = PATH
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
    train_dataset = raw_train.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    test_dataset = raw_test.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    train_dataset.save_to_disk(cache_file_train) if save_dataset is True else None
    test_dataset.save_to_disk(cache_file_test) if save_dataset is True else None
    return train_dataset, test_dataset

def main():
    token = ""
    log_dir = os.path.join('D:/newmodel/output/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer("D:/newmodel/mod5/tokenizer.json") 

    extract_args = {
        "waveform": False,
        "spec": False,
        "pitch_tokens": False,
        "pitch": True,
        "harmonics": False,
        "aperiodics": False,
        "phase_mod": False,
        "crepe": False,        
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
        "dummy": False,
    }

    param = Dimensions(tokens=40000, mels=128, ctx=2048, dims=512, head=4, layer=4, act="gelu", norm_type="rmsnorm")

    train_dataset, test_dataset = prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=extract_args, max_ctx=param.ctx)

    model = Model(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    from functools import partial
    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1000,
        eval_steps=100,
        save_steps=1000,
        warmup_steps=10,
        logging_steps=10,
        logging_dir=log_dir,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="no",
        report_to=["tensorboard"],
        push_to_hub=False,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=False,
        disable_tqdm=False,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        learning_rate=0.00025,
        weight_decay=0.025,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-9, weight_decay=training_args.weight_decay, betas=(0.9, 0.999), amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)
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
    print(model)
    trainer.train()

if __name__ == "__main__":
    main()
