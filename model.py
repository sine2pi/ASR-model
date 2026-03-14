import os, torch, numpy as np
from torch.nn.functional import embedding, scaled_dot_product_attention as SDPA
from torch import nn, Tensor, einsum
from typing import Iterable
from torch.nn.utils.parametrizations import weight_norm
from functools import partial
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from optimizer import FAMScheduler2, MaxFactor, MaxFactorA1
from torch.utils.data import DataLoader    
from datetime import datetime
from essentials import *
from newskip import HybridMyelinatedBlock as skip_layer

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
    tscales = torch.exp(-torch.log(torch.tensor(float(theta), requires_grad=False)) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32, requires_grad=False))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1), requires_grad=False)
    return positional_embedding    

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, layer, act, n_type, norm=False, enc=False):
        super().__init__()

        n.norm = get_norm(n_type, dims) if norm else nn.Identity()
        n.local_norm = get_norm("localnorm", dims) if norm else nn.Identity()        # lets think about this 

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
            n.encoder.append(
                nn.Sequential(
                    act_fn, 
                    weight_norm(nn.Conv1d(dims, dims, kernel_size=3, padding=1)),
                    n.norm,
                    act_fn,
                    nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), 
                    act_fn,
                    nn.Dropout(0.1)
                ))

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

        #figure out what one works best
        # m = n.lin(xa)
        # m = torch.sigmoid(n.lin(xa)) ** t
        # m = torch.sigmoid(n.lin(xa)) 
        
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

    def forward(n, x, xa=None, mask=None, pt=None, skip=False, pattern=None, window=3, zero=False): 
        
        p = pt["pt"] if pt["pt"] is not None else None #need to fix its getting confusing
        b, c, d = x.shape
        k, v = n.kv(aorb(xa, x))
        q = n.q(x)

        if zero:
            if self.rbf:
                qk = self.rbf_scores(q * n.scale, k * n.scale, rbf_sigma=1.0, rbf_ratio=0.3)
            if self.use_pbias:
                pbias = pitch_bias(xa) 
                if pbias is not None:
                    qk = qk + pbias[:,:,:q,:q]

            token_ids = k[:, :, :, 0]
            zscale = torch.ones_like(token_ids)
            fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
            zscale[token_ids.float() == self.pad_token] = fzero
            
            if mask is not None:
                if mask.dim() == 4:
                    mask = mask[0, 0]
                mask = mask[:q, :k] if xa is not None else mask[:q, :q]
                qk = qk + mask * zscale.unsqueeze(-2).expand(qk.shape)

            qk = qk * zscale.unsqueeze(-2)
            w = F.softmax(qk, dim=-1).to(q.dtype)
            wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        if p is not None:
            c = n.c(p)
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

class router(nn.Module):         # eeny meeny miny moe?
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
        n.ln = get_norm(n_type="adalayernorm", dims=dims)

        n.act_fn = get_activation(act)
        n.audio = lambda length, dims: sinusoids(length, dims, THETA)
        
        n.attn = attention(dims, head, layer, n_type=n_type)
        n.router = router(dims, num_types=num_types)
        n.jump = skip_layer(dims, head, layer)

        n.mlp = nn.Sequential(n.ln, tgate(dims, num_types=num_types), 
                              nn.Linear(dims, dims*num_types), get_activation(act), nn.Linear(dims*num_types, dims), n.ln)

    def forward(n, x, xa=None, mask=None, pt=None):

        x  = n.jump(n.ln(x), xa, mask)
        x = n.router(*[x for _ in range(n.layer)]) + n.attn(n.ln(x), mask=mask, pt=pt)
        if xa is not None:
            xa = xa + n.audio(xa.shape[1], xa.shape[-1]).to(device, dtype)
            xa = n.jump(n.ln (xa))
            x = x + n.attn(n.ln(x), xa=n.router(*[xa for _ in range(n.layer)]), pt=pt)

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
       
    def quantize_pitch(n, xa = None, pt = None, num_bins = 256, v_min = -2.0, v_max = 2.0, embedding=False):
        # this isnt working
        indices = ((pt - v_min) / (v_max - v_min) * (num_bins - 1)).round()
        tensor = torch.clamp(indices, 0, num_bins - 1)
        tensor = torch.polar(xa, tensor) if xa is not None else tensor
        tensor = torch.view_as_real(tensor) if xa is not None else tensor
        return nn.Embedding(tensor, n.dims) if embedding else tensor

    def forward(n, x, xa=None, seq=False) -> Tensor:
        blend = torch.sigmoid(n.blend)
        mask = n.mask[:x.shape[1], :x.shape[1]]

        x1 = n.token(x)    

        if xa['pt'] is not None:
            pt = n.quantize_pitch(pt=xa['pt'])
            x2 = n.pitch_tokens(pt)
            x1 = x1 + x2 
        else:
            pt = xa #  lets pass the dict im thinking of using the raw features for something downstream

        x = (x1 + n.position[:x.shape[-1]]).to(device, dtype)

        for i in n.block:
            a = i(x, mask=mask, pt=pt)
            b = i(a, xa=i(xa['a']), pt=pt)
            c = i(b, xa=i(xa['b']), pt=pt)
            d = i(c, xa=i(xa['c']), pt=pt)

            # for j in [(xa), (xb), (xc)]: e = i(x, xa=i(j, pt=pt))
            # e = torch.mean(torch.stack([xa, xb, xc]), dim=0)

            e = a + b + c
            f = torch.cat([d, e], dim=1)
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

    @torch.no_grad()
    def generate(n, spectrogram=None, pitch=None, waveform=None, pitch_tokens=None, max_new_tokens=150):
        n.eval()
        fx = next((t for t in (pitch, spectrogram, waveform) if t is not None), None)

        xa_dict = TensorDict({
            'a': aborc(pitch, spectrogram, waveform),
            'b': aborc(spectrogram, pitch, waveform),
            'c': aborc(waveform, pitch, spectrogram),
        }, batch_size=fx.shape[0]).to(device)
        
        xa_enc = n.enc(no_none(xa_dict))
        if pitch_tokens is not None:
            xa_enc['pt'] = pitch_tokens 

        y = torch.tensor([[1]], dtype=torch.long, device=device).repeat(fx.shape[0], 1)
        
        for _ in range(max_new_tokens):
            logits = n.processor(y, xa_enc.clone(), seq=True) 
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            y = torch.cat((y, next_token), dim=1)
            if (next_token == 2).all():
                break
                
        return y

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

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    metadata_file = "./lb1/metadata.csv"
    data_dir = "./lb1"

    # metadata_file = "./LJSpeech1000/metadata.csv"
    # data_dir = "./LJSpeech1000"
    # metadata_file = "./cv17_1000/metadata.csv"
    # data_dir = "./cv17_1000"

    log_dir = os.path.join('./logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("./tokenizer.json") 
    
    extract_args = {

        "spectrogram": False,
        "pitch": True,
        "waveform": False,
        "pitch_tokens": False,
        "harmonics": False,
        "aperiodic": False,
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

    # We group params so that 'jump' (HybridMyelinatedBlock) uses bias=2 (Median),
    # and all standard logic blocks use bias=1 (Max)
    main_params = []
    jump_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'jump' in name or 'policy_net' in name or 'micro_filter' in name:
            jump_params.append(p)
        else:
            main_params.append(p)

    optimizer = MaxFactor([
        {'params': main_params, 'bias': 1.0}, # Max updates for core features
        {'params': jump_params, 'bias': 2.0}  # Median updates for routing logic to prevent wild jumps
    ], lr=0.025, beta_decay=-0.8, eps=(1e-8, 1e-3), d=1.0, w_decay=0.01, gamma=0.99, max=False, clip=False, cap=0.0)

    scheduler = FAMScheduler2(optimizer, warmup_steps=10, total_steps=100, 
                 decay_start_step=None, warmup_start_lr=1e-6, eta_min=1e-6, last_epoch=-1) 

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
        eval_interval=0,
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
