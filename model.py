import os, random, warnings, logging, time
import torch, numpy as np
from datasets import load_dataset, Audio
from torch.nn.functional import scaled_dot_product_attention
from torch import nn, Tensor
from typing import Iterable, Optional, Any, List, Dict
from functools import partial
from datetime import datetime
from tensordict import TensorDict
from dataclasses import dataclass
from torch.utils.data import DataLoader    
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

torch.backends.cudnn.conv.fp32_precision = 'tf32'

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

PATH = 'H:/cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
THETA = 30000.0

@dataclass
class Dimensions:
    tokens: int
    mels: int
    dims: int
    head: int
    layer: int
    act: str
    n_type: str

def l2norm(t):
    return torch.nn.functional.normalize(t, dim = -1)

def Sequential(*modules):
    return nn.Sequential(*filter(have, modules))  

def have(a):
    return a is not None  

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, b) if not have(c) else c

def no_none(xa):
    return xa.apply(lambda tensor: tensor if tensor is not None else None)

def shift_right(x):
    return torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def sinusoids(ctx, dims, theta=THETA):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding    

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False, enc=False):
        super().__init__()
        
        act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1),
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        theta = nn.Parameter(torch.tensor(THETA), requires_grad=True)
        n.audio = lambda length, dims: sinusoids(length, dims, theta)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(n_type, dims) if norm else nn.Identity()

    def _process_feature(n, x):
        # x = x.to(device, dtype)
        # x = shift_right(x) if n.shift else x
        if x.dim() == 2:
            x = x.unsqueeze(0)   
        if x.shape[1] > 1:      
            x = n.conv1(x)
        else:
            x = n.conv2(x)
        x = n.encoder(x).permute(0, 2, 1).contiguous().to(device, dtype)
        x = x + n.audio(x.shape[1], x.shape[-1]).to(device, dtype)
        x = n.ln(x)
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
        n._head = nn.Linear(dims, 1) 
    
    def _compute_freqs(n, x=None, mask=None):
        if mask is None:
            scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
            return x.mean(dim=-1) * scale / 1000 if x is not None else 200 * scale / 1000
        else: 

            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype) / n.head_dim * torch.log(torch.tensor(x.mean(dim=-1) * THETA if x is not None else THETA))

    def forward(n, x=None, xa=None, mask=None): 
        t = torch.arange(x.shape[2], device=device, dtype=dtype).float()
        freqs = torch.einsum('i,j->ij', t,  n._compute_freqs(mask=mask))
        
        if xa is not None:
            freqs = torch.polar(xa.mean(dim=-1).unsqueeze(-1), freqs)
        else:   
            freqs = torch.polar(torch.ones_like(freqs), freqs)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        out = torch.cat([x1.type_as(x), x2], dim=-1)
        return out
    
class OneShot(nn.Module):
    def __init__(n, dims: int, head: int, scale: float = 0.3, features: Optional[List[str]] = None):
        super().__init__()
        if features is None:    
            features = ["spectrogram", "waveform", "pitch", "aperiodic", "harmonics"]
        n.head = head
        n.head_dim = dims // head
        n.scale = 1.0 // len(features) if features else scale

        n.q = nn.Linear(dims, dims)
        n.k = nn.Linear(dims, dims)

    def forward(n, x: Tensor, xa: Tensor, feature=None) -> Tensor | None:
        B, L, D = x.shape
        K = xa.size(1)
        q = n.q(x).view(B, L, n.head, n.head_dim).transpose(1,2)
        k = n.k(xa).view(B, K, n.head, n.head_dim).transpose(1,2)
        bias = (q @ k.transpose(-1, -2)) * n.scale / math.sqrt(n.head_dim)
        return bias

class curiosity(nn.Module):
    def __init__(n, d, h, bias=True):
        super().__init__()
        n.h  = h
        n.dh = d // h
        n.qkv = nn.Linear(d, d * 3, bias=bias)
        n.qkv_aux = nn.Linear(d, d * 3, bias=bias)
        n.o  = nn.Linear(d, d, bias=bias)
        n.g  = nn.Parameter(torch.zeros(h))

    def split(n, x):
        b, t, _ = x.shape
        return x.view(b, t, n.h, n.dh).transpose(1, 2)

    def merge(n, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(n, x, xa, mask=None):

        q, k, v   = n.qkv(x).chunk(3, -1)
        qa, ka, va = n.qkv_aux(xa).chunk(3, -1)
        q, k, v   = map(n.split, (q, k, v))
        qa, ka, va = map(n.split, (qa, ka, va))
        dots      = (q @ k.transpose(-2, -1)) / n.dh**0.5
        dots_aux  = (q @ ka.transpose(-2, -1)) / n.dh**0.5
        if mask is not None: dots = dots.masked_fill(mask, -9e15)
        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)
        h_main = p  @ v
        h_aux  = pa @ va
        g = torch.sigmoid(n.g).view(1, -1, 1, 1)
        out = n.merge(h_main * (1 - g) + h_aux * g)
        return n.o(out)

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
        q, k = n.rot(q, xa=None, mask=mask), n.rot(k, xa=None, mask=mask)  

        if skip and not have(p): 
            a = scaled_dot_product_attention(n.ln(q), n.ln(k[:, :, ::max(1, 6 - n.layer), :]), v[:, :, ::max(1, 6 - n.layer), :], is_causal=have(mask))
            
        elif have(p) and p > 1: 
            k, v = k[:, :, ::p, :], v[:, :, ::p, :]
            a = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=have(mask))
        else: 
            a = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=have(mask))

        if n.modal and xa is not None:

            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb = n.q(x), n.q(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)

            if have(p) and p > 1:
                k, ka, kb = k[:, :, ::p, :], k[:, :, 1::p, :], k[:, :, 2::p, :]
                v, va, vb = v[:, :, ::p, :], v[:, :, 1::p, :], v[:, :, 2::p, :]
            elif skip:
                ka, va = ka[:, :, ::max(1, 6 - n.layer), :], va[:, :, ::max(1, 6 - n.layer), :]
                kb, vb = kb[:, :, ::max(1, 6 - n.layer), :], vb[:, :, ::max(1, 6 - n.layer), :]
            else:
                ka, va = ka, va
                kb, vb = kb, vb
                
            b = scaled_dot_product_attention(n.ln(qa), n.ln(kb), vb, is_causal=have(mask))
            c = scaled_dot_product_attention(n.ln(qb), n.ln(ka), va, is_causal=have(mask))

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
        
    def forward_triplet_toy(n, x, xa=None, mask=None, pt=None, context_window=3):
        q = n.q(x)
        k, v = n.kv(aorb(xa, x))
        b, h, seq_len, d = q.shape 
        scale = d ** -0.5

        if pt is not None: c = n.c(pt)
        else: c = torch.zeros_like(x)

        triplet_scores = torch.zeros(b, h, seq_len, seq_len, device=device)

        for i in range(seq_len):
            for j in range(seq_len):
                context_start = max(0, min(i, j) - context_window)
                context_end = min(seq_len, max(i, j) + context_window)
                
                for k in range(context_start, context_end): 
                    score = (q[:, :, i, :] * k[:, :, j, :] * c[:, :, k, :]).sum(dim=-1)
                    triplet_scores[:, :, i, j] += score

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale + triplet_scores

        if have(mask): qk = qk + mask[:seq_len, :seq_len]

        qk = torch.nn.functional.softmax(qk, dim=-1)
        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        return n.out(wv)

class gate(nn.Module):
    def __init__(n, dims, num_feature):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_feature)])
        n.features = nn.Sequential(nn.Linear(dims, num_feature), nn.Softmax(dim=-1))
        n.top = nn.Linear(dims, num_feature)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x, num=None):
        types, indices = torch.topk(n.top(x), num, dim=-1)
        type = torch.zeros_like(n.features(x))
        type.scatter_(-1, indices, torch.nn.functional.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, skip=False, pattern=None):
        super().__init__()

        n.head = head
        n.skip = skip
        n.expand = head
        n.ln = get_norm(n_type=n_type, dims=dims)
        n.act_fn = get_activation(act)

        n.attn = attention(dims, head, layer, n_type=n_type)
        n.gate = gate(dims, num_feature=n.expand) if not skip else nn.Identity()
        n.mlp = nn.Sequential(n.ln, nn.Linear(dims, dims*n.expand), get_activation(act), nn.Linear(dims*n.expand, dims))

    def forward(n, x, xa=None, mask=None, pt=None, skip=False, pattern=None):
        x = x + n.attn(n.ln(x), mask=mask, pt=pt, skip=skip, pattern=pattern)[0]
        if xa is not None: 
            xa = xa + n.gate(xa, n.expand // 2)
            x = x + n.attn(n.ln(x), xa=xa, pt=pt, skip=skip, pattern=pattern)[0]
        return x + n.mlp(x)

class attn_pass(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, skip=True, pattern=None):
        super().__init__()
        
        n.layers = nn.ModuleList()
        for i in range(layer): n.layers.append(residual(dims, head, layer, act, n_type, skip=skip and i in skip, pattern=pattern[i] if pattern else None))

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
            [residual(dims, head, layer, act, n_type, skip=False, pattern=None) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, seq=False) -> Tensor:

        xa, xb, xc, pt = (xa.pop(k, None) for k in ('a', 'b', 'c', 'pt')) if isinstance(xa, TensorDict) else (None, None, None, None)

        blend = torch.sigmoid(n.blend)
        x = (n.token(x) + n.position[:x.shape[-1]]).to(device, dtype)

        for i in n.block:
            a = i(x,  mask=n.mask, pt=pt)
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
        xa['pt']  = pitch_tokens if pitch_tokens is not None else None

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
    
########## pipeline functions ##########

def prepare_datasets(tokenizer, sample_rate, streaming, load_saved, save_dataset, cache_dir, extract_args, max_ctx, fleurs):
    from datasets import load_dataset, Audio
    token = "" # hugging face datasets

    if load_saved:
        if cache_dir is None:
            cache_dir = PATH
        else:
            cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        train = os.path.join(cache_dir, "train.arrow")
        test = os.path.join(cache_dir, "test.arrow")

        if os.path.exists(train) and os.path.exists(test):
            from datasets import Dataset
            train = Dataset.load_from_disk(train)
            test = Dataset.load_from_disk(test)
            return train, test   
        
    if fleurs:
        train = load_dataset("google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=False, cache_dir=cache_dir).take(1000).cast_column("audio", Audio(sampling_rate=sample_rate)).map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
        test  = load_dataset("google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=False, cache_dir=cache_dir).take(100).cast_column("audio", Audio(sampling_rate=sample_rate)).map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])

    else:
        train = load_dataset("fixie-ai/common_voice_17_0", "en", split="train", streaming=streaming, token=token, trust_remote_code=True).take(1000).cast_column("audio", Audio(sampling_rate=sample_rate)).map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "sentence"])
        test = load_dataset("fixie-ai/common_voice_17_0", "en", split="test", streaming=streaming, token=token, trust_remote_code=True).take(100).cast_column("audio", Audio(sampling_rate=sample_rate)).map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "sentence"])

    return train, test

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

def track_grad_norms(model):
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm(2).item()
            grad_norms[name] = norm
    return grad_norms

def compute_metrics(pred, tokenizer=None, model=None, print_pred=False, num_samples=0, logits=None, compute_result = False):
    def clean(ids, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], (list, torch.Tensor, np.ndarray)):
            return [[int(q) for q in seq if q not in (-100, pad_token_id, bos_token_id, eos_token_id)] for seq in ids]
        else:
            return [int(q) for q in ids if q not in (-100, pad_token_id, bos_token_id, eos_token_id)]

    if isinstance(pred, dict):
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
    else:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    else:
        pred_ids = pred_ids
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

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
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
        efficiency_score = (100 - wer) / trainable_params if trainable_params > 0 else 0.0
        per_layer_norms = track_grad_norms(model)
        if not isinstance(per_layer_norms, dict) or not per_layer_norms:
            per_layer_norms = {}
    else:
        trainable_params = 0.0
        efficiency_score = 0.0
        per_layer_norms = {}

    result = {
        "wer": float(wer),
        "efficiency_score": float(efficiency_score),
    }

    if isinstance(per_layer_norms, dict):
        for k, v in per_layer_norms.items():
            result[f"per_layer_norms_{k}"] = float(v)
    return result

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
            sp_ids = [tokenizer.token_to_id(t) for t in ["<UNK>, <PAD>", "<BOS>", "<EOS>"]]
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

def get_norm(n_type: str, dims: Optional[int] = None, num_groups: Optional[int] = None)-> nn.Module:

    if n_type in ["batchnorm", "instancenorm"] and dims is None:
        raise ValueError(f"'{n_type}' requires 'dims'.")
    if n_type == "groupnorm" and num_groups is None:
        raise ValueError(f"'{n_type}' requires 'num_groups'.")

    norm_map = {
        "layernorm": lambda: nn.LayerNorm(normalized_shape=dims, bias=False),
        # "instanceRMS": lambda: InstanceRMS(dims=dims),   
        "instancenorm": lambda: nn.InstanceNorm1d(num_features=dims, affine=False, track_running_stats=False),     
        "rmsnorm": lambda: nn.RMSNorm(normalized_shape=dims),        
        "batchnorm": lambda: nn.BatchNorm1d(num_features=dims),
        "instancenorm2d": lambda: nn.InstanceNorm2d(num_features=dims),
        "groupnorm": lambda: nn.GroupNorm(num_groups=num_groups, num_channels=dims),
        }
   
    norm_func = norm_map.get(n_type)
    if norm_func:
        return norm_func()
    else:
        print(f"Warning: Norm type '{n_type}' not found. Returning LayerNorm.")
        return nn.LayerNorm(dims) 

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
    }
    return act_map.get(act, nn.GELU())

def extract_features(batch, tokenizer, spectrogram=False, pitch=False, waveform=False, pitch_tokens=False, hop_length=160, sample_rate=16000, n_fft=1024):
    import pyworld as pw

    mode = "mean"
    dummy_audio = False
    dummy_text = False
    audio = batch["audio"]

    if dummy_text:
        labels = [1] * 32
    else:
        labels = tokenizer.encode(batch["transcription" if "transcription" in batch else "sentence"])

    if dummy_audio:
        if isinstance(audio, str):
            dummy, sample_rate = torchaudio.load(uri=audio, normalize=True, backend="ffmpeg")
        elif isinstance(audio, dict):
            dummy = torch.tensor(data=audio["array"]).float()
        audio = torch.zeros_like(dummy)
    else:
        if isinstance(audio, str):
            audio, sample_rate = torchaudio.load(uri=audio, normalize=True, backend="ffmpeg")
        elif isinstance(audio, dict):
            audio = torch.tensor(data=audio["array"]).float()

    if pitch_tokens:
        wavnp = audio.numpy().astype(np.float64)
        f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)

        audio_duration = len(audio) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        
        f0_tensor = torch.from_numpy(f0_np)
        t_tensor = torch.from_numpy(t)

        mel_f0 = torch.zeros_like(f0_tensor)
        positive_f0_mask = f0_tensor > 0
        mel_f0[positive_f0_mask] = 2595. * torch.log10(1 + f0_tensor[positive_f0_mask] / 700.0)

        start_idx = torch.searchsorted(t_tensor, token_starts, side="left")
        end_idx = torch.searchsorted(t_tensor, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)

        for q in range(T):
            lo, hi = start_idx[q], max(start_idx[q] + 1, end_idx[q])
            segment = mel_f0[lo:hi]
            
            voiced_segment = segment[segment > 0]
            
            if len(voiced_segment) > 0:
                if mode == "mean":
                    pitch_tok[q] = voiced_segment.mean()
                elif mode == "median":
                    pitch_tok[q] = torch.median(voiced_segment)
                else:
                    pitch_tok[q] = voiced_segment[-1]

        mean_pitch = pitch_tok[pitch_tok > 0].mean() if (pitch_tok > 0).any() else 0.0
        std_pitch = pitch_tok[pitch_tok > 0].std() if (pitch_tok > 0).any() else 1.0
        pt_tensor = (pitch_tok - mean_pitch) / (std_pitch + 1e-6)

        if pt_tensor.numel() > 0:
            bos_pitch = pt_tensor.item() if pt_tensor.numel() > 0 else 0.0
        else:
            bos_pitch = 0.0

        bos_tensor = torch.tensor([bos_pitch], dtype=pt_tensor.dtype)
        pt_tensor = torch.cat([bos_tensor, pt_tensor])

    if pitch:
        f0, t = pw.dio(audio.numpy().astype(np.float64), sample_rate, frame_period=hop_length / sample_rate * 1000)
        f0 = pw.stonemask(audio.numpy().astype(np.float64), f0, t, sample_rate)
        p_tensor = torch.from_numpy(f0)

    if spectrogram:
        spectrogram_config = {
            "hop_length": hop_length,
            "f_min": 50,
            "f_max": 1000,
            "n_mels": 128,
            "n_fft": n_fft,
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
        mel_spectrogram = transform(audio)
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        s_tensor = (log_mel + 4.0) / 4.0

    if waveform:
        current = audio.shape[-1]  
        target = int((len(audio) / sample_rate) * exact_div(sample_rate, hop_length))
            
        if audio.dim() == 1:
            aud = audio.unsqueeze(0).unsqueeze(0) 
        elif audio.dim() == 2:
            aud = audio.unsqueeze(0)  

        if current > target:
            w_tensor = torch.nn.functional.adaptive_avg_pool1d(aud, target)
        else:
            w_tensor = torch.nn.functional.interpolate(aud, size=target, mode='linear', align_corners=False)

        if w_tensor.dim() == 3 and w_tensor.shape[0] == 1:
            w_tensor = w_tensor.squeeze(0)
        else:
            w_tensor = w_tensor

    return {
        "waveform": w_tensor if waveform else None,
        "spectrogram": s_tensor if spectrogram else None,
        "pitch_tokens": pt_tensor if pitch_tokens else None,
        "pitch": p_tensor if pitch else None,
        "labels": labels if labels is not None else None,
    }

@dataclass
class DataCollator:
    tokenizer: Any
    def __call__(n, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())

        batch = {}
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
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
                batch["text_ids"] = torch.tensor(all_ids, dtype=torch.long)
                batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

            elif key in ["spectrogram", "waveform", "pitch", "pitch_tokens"]:
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
                        pad_item = torch.nn.functional.pad(item, (0, pad_width), mode='constant', value=pad_token_id)
                    else:
                        pad_item = item
                    padded.append(pad_item)
                batch[key] = torch.stack(padded)
        return batch

def train_and_evaluate(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    loss_fn,
    metric_fn,
    max_steps=1000,
    device="cuda",
    acc_steps=1,
    clear_cache=False,
    log_interval=10,
    eval_interval=100,
    save_interval=1000,
    checkpoint_dir="checkpoint_dir",
    log_dir="log_dir",
):

    logging.basicConfig(level=logging.INFO)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    
    oneshot_modules = [m for m in model.modules() if isinstance(m, OneShot)]
    grad_history = []
    
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

    progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour="green")
    progress_bar.update(global_step)
    model.train()
    optimizer.zero_grad()

    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            dataset_epochs += 1
            print(f"Starting dataset epoch {dataset_epochs}")

            if step_in_report > 0:
                avg_loss = total_loss / step_in_report
                logging.info(
                    f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}"
                )
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        features = TensorDict({k: v.to(device) for k, v in batch.items() if k in ["spectrogram","waveform","pitch", "pitch_tokens"] and v is not None}, batch_size=batch["text_ids"].shape[0]).to(device)
        text_ids = batch["text_ids"].to(device)
        labels = batch["labels"].long().to(device)

        output = model(text_ids=text_ids, labels=labels, spectrogram=features.get("spectrogram"), pitch=features.get("pitch"), waveform=features.get("waveform"), pitch_tokens=features.get("pitch_tokens"))
        loss = output["loss"]

        total_loss += loss.item()
        loss = loss / acc_steps

        scaler.scale(loss).backward()

        per_layer_norms = track_grad_norms(model)
        for k, v in per_layer_norms.items():
            writer.add_scalar(f"train/per_layer_norms_{k}", v, global_step)

        if (global_step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            grad_history.append(total_norm)
            
            if len(grad_history) > 10:
                recent_avg = sum(grad_history[-5:]) / 5
                prev_avg = sum(grad_history[-10:-5]) / 5
                
                for module in oneshot_modules:
                    if recent_avg > prev_avg * 1.2:
                        module.scale *= 0.9
                        logging.info(f"Reducing OneShot scale to {module.scale:.4f} (grad norm: {total_norm:.2f})")
                    elif recent_avg < prev_avg * 0.8:
                        module.scale *= 1.1
                        logging.info(f"Increasing OneShot scale to {module.scale:.4f} (grad norm: {total_norm:.2f})")
                
                if len(grad_history) > 100:
                    grad_history = grad_history[-100:]

            if global_step % log_interval == 0:
                writer.add_scalar("GradNorm", total_norm, global_step)
                if oneshot_modules:
                    writer.add_scalar("OneShot/scale", oneshot_modules[0].scale, global_step)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch["text_ids"]) / (end_time - start_time)

        if global_step % log_interval == 0:
            writer.add_scalar(
                tag="Loss/train",
                scalar_value=total_loss / (global_step + 1),
                global_step=global_step,
            )
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar(
                tag="LearningRate", scalar_value=lr, global_step=global_step
            )
            writer.add_scalar(
                tag="SamplesPerSec",
                scalar_value=samples_per_sec,
                global_step=global_step,
            )

        if global_step % eval_interval == 0:
            model.eval()
            start = time.time()
            eval_loss = 0
            all_p = []
            all_l = []
            batch = 0
            total = 0
            predictions = {}
            
            with torch.no_grad():
                for eval_batch in eval_loader:
                    features = TensorDict({k: v.to(device) for k, v in eval_batch.items() if k in ["spectrogram","waveform","pitch", "pitch_tokens"] and v is not None}, batch_size=eval_batch["text_ids"].shape[0]).to(device)
                    text_ids = eval_batch["text_ids"].to(device)
                    labels = eval_batch["labels"].long().to(device)

                    batch_size = text_ids.size(0)
                    total += batch_size

                    output = model(text_ids=text_ids, labels=labels, spectrogram=features.get("spectrogram"), pitch=features.get("pitch"), waveform=features.get("waveform"), pitch_tokens=features.get("pitch_tokens"))
                    loss = output["loss"]
                    eval_loss += loss.item()
                    all_p.extend(
                        torch.argmax(output["logits"], dim=-1).cpu().numpy().tolist()
                    )
                    all_l.extend(labels.cpu().numpy().tolist())
                    batch += 1

            eval_time = time.time() - start
            loss_avg = eval_loss / batch if batch > 0 else 0

            p = {
                "predictions": np.array(all_p, dtype=object),
                "label_ids": np.array(all_l, dtype=object),
            }
            metrics = metric_fn(p, tokenizer=tokenizer, model=model)
            writer.add_scalar("Loss/eval", loss_avg, global_step)
            writer.add_scalar("WER", metrics["wer"], global_step)
            writer.add_scalar("EvalSamples", total, global_step)
            writer.add_scalar("EvalTimeSeconds", eval_time, global_step)
      
            if "per_layer_norms" in metrics:
                for layer, norm in metrics["per_layer_norms"].items():
                    writer.add_scalar(f"eval/per_layer_norms/{layer}", norm, global_step)

            if oneshot_modules:
                writer.add_scalar("OneShot/eval_scale", oneshot_modules[0].scale, global_step)
            
            lr = scheduler.get_last_lr()[0]

            print(
                f"• STEP:{global_step} • samp:{samples_per_sec:.1f} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}"
                + (f" • OneShot:{oneshot_modules[0].scale:.4f}" if oneshot_modules else "")
            )

            logging.info(
                f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}"
                + (f", OneShot: {oneshot_modules[0].scale:.4f}" if oneshot_modules else "")
            )
            model.train()

        if global_step > 0 and global_step % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_step_{global_step}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        global_step += 1
        step_in_report += 1

        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {
            "loss": f"{avg_loss:.4f}",
            "lr": f"{lr:.6f}",
            "samp": f"{samples_per_sec:.1f}",
        }   
        if oneshot_modules:
            postfix_dict["os"] = f"{oneshot_modules[0].scale:.4f}"
            
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)

    path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), path)
    print(f"Training completed after {global_step} steps. Final model saved to {path}")
    writer.close()
    progress_bar.close()
    return model

def main():
    
    log_dir = os.path.join('./ignore/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("./tokenizer.json") 
    
    extract_args = {

        "spectrogram": False,
        "pitch": True,
        "waveform": False,
        "pitch_tokens": False,
        "hop_length": 160,
        "sample_rate": 16000
    }

    param = Dimensions(tokens=40000, mels=128, dims=512, head=4, layer=4, act="gelu", n_type="layernorm")

    train_dataset, test_dataset = prepare_datasets(tokenizer, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=PATH, extract_args=extract_args, max_ctx=2048, fleurs=True)

    model = Model(param).to('cuda')

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)
    Collator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=1, collate_fn=Collator, num_workers=0
    )

    eval_dataloader = DataLoader(
        dataset=test_dataset, batch_size=1, collate_fn=Collator, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, eps=1e-9, weight_decay=0.025, betas=(0.9, 0.999), amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-9, last_epoch=-1)

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
        max_steps=1000,
        device="cuda",
        acc_steps=1,
        clear_cache=False,
        log_interval=10,
        eval_interval=100,
        save_interval=1000,
        checkpoint_dir=log_dir,
        log_dir=log_dir,
    )

if __name__ == "__main__":
    main()
