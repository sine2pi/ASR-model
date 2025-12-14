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

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, n_type, norm=False, enc=False):
        super().__init__()
        
        act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1),
            act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(n_type, dims) if norm else nn.Identity()

    def _process_feature(n, x):
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

    def _compute_freqs(n, x=None, mask=None):
        if mask is None:
            scale = n.gammatone()
            # scale = n.wideband()
            # scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype, requires_grad=False) / 2595) - 1

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

        k, v = n.kv(aorb(xa, x))
        q = n.q(x)
        q, k = n.rot(q, xa=x, mask=mask), n.rot(k, xa=x if xa is None else xa, mask=mask)  

        if skip and not have(p): 
            a = SDPA(n.ln(q), n.ln(k[:, :, ::max(1, 6 - n.layer), :]), v[:, :, ::max(1, 6 - n.layer), :], is_causal=have(mask))
            
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
            elif skip:
                ka, va = ka[:, :, ::max(1, 6 - n.layer), :], va[:, :, ::max(1, 6 - n.layer), :]
                kb, vb = kb[:, :, ::max(1, 6 - n.layer), :], vb[:, :, ::max(1, 6 - n.layer), :]
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

# class tgate(nn.Module):
#     def __init__(n, dims, num_types=2):
#         super().__init__()

#         n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
#         n.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))

#     def forward(n, x):
#         types = n.classifier(x)
#         gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
#         return  torch.sum(gates * types.unsqueeze(2), dim=-1)

class gate(nn.Module):
    def __init__(n, dims, num_types):
        super().__init__()

        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_types)])
        n.features = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
        n.top = nn.Linear(dims, num_types)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x, num=2):
        types, indices = torch.topk(n.top(x), num, dim=-1)
        type = torch.zeros_like(n.features(x))
        type.scatter_(-1, indices, torch.nn.functional.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class Router(nn.Module):
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
    def __init__(n, dims, head, layer, act, n_type, num_types=8):
        super().__init__()

        n.expand = head
        n.ln = get_norm(n_type=n_type, dims=dims)
        n.act_fn = get_activation(act)
        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        
        n.attn = attention(dims, head, layer, n_type=n_type)
        n.router = Router(dims, num_types=3)
        n.mlp = nn.Sequential(n.ln, nn.Linear(dims, dims*num_types), get_activation(act), nn.Linear(dims*num_types, dims))

    def forward(n, x, xa=None, mask=None, pt=None):
        x = x + n.attn(n.ln(x), mask=mask, pt=pt)
        if xa is not None: 
            xa = xa + n.audio(xa.shape[1], xa.shape[-1]).to(device, dtype)
            xa = n.router(*[xa for _ in range(3)])  
            x = x + n.attn(n.ln(x), xa=xa, pt=pt)
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

        # n.block_main: Iterable[residual] = nn.ModuleList(
        #     [residual(dims, head, layer, act, n_type) for _ in range(layer)]) 

        n.block: Iterable[residual] = nn.ModuleList(
            [residual(dims, head, layer, act, n_type) for _ in range(layer)]) 

        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, seq=False) -> Tensor:

        m = {}
        if isinstance(xa, TensorDict):
            m = {k: v for k, v in xa.items() if k in ('a', 'b', 'c', 'pt')}
        pt = m.pop('pt', None)

        blend = torch.sigmoid(n.blend)
        x = (n.token(x) + n.position[:x.shape[-1]]).to(device, dtype)

        for i, b in enumerate(n.block):
            x = b(x, mask=n.mask, pt=pt)
            if 'a' in m: x = b(x=x, xa=b(m['a'], pt=pt), pt=pt)
            if 'b' in m: x = b(x=x, xa=b(m['b'], pt=pt), pt=pt)
            if 'c' in m: x = b(x=x, xa=b(m['c'], pt=pt), pt=pt)

            # for j in [(m['a']), (m['b']), (m['c'])]: e = b(x, xa=b(j, pt=pt))
            e = torch.mean(torch.stack([m['a'], m['b'], m['c']]), dim=0)
            f = torch.cat([x, e], dim=1)
            g = b(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:])
            x = g if seq else blend * (x) + (1 - blend) * g 

        # xa, xb, xc, pt = (xa.pop(k, None) for k in ('a', 'b', 'c', 'pt')) if isinstance(xa, TensorDict) else (None, None, None, None)

        # blend = torch.sigmoid(n.blend)
        # x = (n.token(x) + n.position[:x.shape[-1]]).to(device, dtype)

        # for i in n.block:
        #     a = i(x, mask=n.mask, pt=pt)
        #     b = i(a, xa=i(xa, pt=pt))
        #     c = i(b, xa=i(xb, pt=pt))
        #     d = i(c, xa=i(xc, pt=pt))

        #     for j in [(xa), (xb), (xc)]: e = i(x, xa=i(j, pt=pt))

        #     f = torch.cat([d, e], dim=1)
        #     g = i(x=f[:, :x.shape[1]], xa=f[:, x.shape[1]:])
        #     x = g if seq else blend * (d) + (1 - blend) * g 

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
    
def main():

    metadata_file = "./cv17_1000/metadata.csv"
    data_dir = "./cv17_1000"

    log_dir = os.path.join('./logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("./tokenizer.json") 
    
    extract_args = {

        "spectrogram": False,
        "pitch": True,
        "waveform": False,
        "pitch_tokens": False,
        "hop_length": 160,
        "sample_rate": 16000,
        "mels": 128
    }

    param = Dimensions(tokens=40000, mels=128, dims=512, head=4, layer=4, act="gelu", n_type="layernorm")

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

    optimizer = MF(model.parameters(), lr=0.025, beta_decay=-0.8, eps=(1e-10, 1e-3), d=1.0, w_decay=0.01, gamma=0.99, max=False)
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

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
