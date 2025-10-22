import os, random, warnings, logging
import torch, numpy as np
from datasets import load_dataset, Audio
from torch.nn.functional import scaled_dot_product_attention
from torch import nn, Tensor
from typing import Iterable
from functools import partial
from datetime import datetime
from tensordict import TensorDict
from dataclasses import dataclass
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from einops.layers.torch import Rearrange
from echoutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

torch.backends.cudnn.conv.fp32_precision = 'tf32'

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

THETA = 30000.0

def l2norm(t):
    return F.normalize(t, dim = -1)

def have(a):
    return a is not None
    
def Sequential(*modules):
    return nn.Sequential(*filter(have, modules))    

def aorb(a, b):
    return a if have(a) else b

def no_none(xa):
    return xa.apply(lambda tensor: tensor if tensor is not None else None)

@dataclass
class Dimensions:
    tokens: int
    mels: int
    dims: int
    head: int
    layer: int
    act: str
    n_type: str

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
    
    def _compute_freqs(n, x=None, mask=None): # mask flags causal attention x
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

        if skip and not have(p): a = scaled_dot_product_attention(n.ln(q), n.ln(k[:, :, ::max(1, 6 - n.layer), :]), v[:, :, ::max(1, 6 - n.layer), :], is_causal=mask is not None)
            
        elif have(p) and p > 1: 
            k, v = k[:, :, ::p, :], v[:, :, ::p, :]
            a = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=mask is not None)
        else: 
            a = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=mask is not None)

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
                
            b = scaled_dot_product_attention(n.ln(qa), n.ln(kb), vb, is_causal=mask is not None)
            c = scaled_dot_product_attention(n.ln(qb), n.ln(ka), va, is_causal=mask is not None)

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

        qk = F.softmax(qk, dim=-1)
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
        type.scatter_(-1, indices, F.softmax(types, dim=-1))
        features = torch.sigmoid(n.alpha) * type + (1 - torch.sigmoid(n.alpha)) * n.features(x)
        return torch.sum(torch.stack([gate(x) for gate in n.gates], dim=-1) * features.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, n_type, expand=4, skip=False, pattern=None):
        super().__init__()

        n.head = head
        n.skip = skip
        n.ln = get_norm(n_type=n_type, dims=dims)
        n.act_fn = get_activation(act)

        n.attn = attention(dims, head, layer, n_type=n_type)
        n.gate = gate(dims, num_feature=head) if not skip else nn.Identity()
        n.mlp = nn.Sequential(n.ln, nn.Linear(dims, dims*expand), get_activation(act), nn.Linear(dims*expand, dims))

    def forward(n, x, xa=None, mask=None, pt=None, skip=False, pattern=None):

        x = x + n.attn(n.ln(x), mask=mask, pt=pt, skip=skip, pattern=pattern)[0]
        if xa is not None: 
            xa = xa + n.gate(xa, n.head // 2)
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
            [residual(dims=dims, head=head, layer=layer, act=act, n_type=n_type) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, seq=False) -> Tensor:

        pt, xb, xc, xa = (xa.pop(k, None) for k in ('pt', 'b', 'c', 'a')) if isinstance(xa, TensorDict) else (None, None, None, None)

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
        # n.pas = attn_pass(param.dims, param.head, param.layer, param.act, param.n_type)

        n.layer = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer += 1        

    def forward(n, labels=None, input_ids=None, spectrogram=None, pitch=None, waveform=None, 
            harmonics=None, aperiodics=None, phase=None, hilbert=None, pitch_tokens=None):

        fx = next((t for t in (pitch, spectrogram, waveform) if t is not None), None)
        xa = TensorDict({
            'a': aorb(pitch, fx),
            'b': aorb(spectrogram, fx),
            'c': aorb(waveform, fx),
            'd': harmonics,
            'e': aperiodics,
            'f': phase,
            'g': hilbert,
            # "pt": pitch_tokens,
            }, batch_size=fx.shape[0])

        x = input_ids
        xa = n.enc(no_none(xa))
        xa['pt']  = pitch_tokens if pitch_tokens is not None else None

        output = n.processor(x, xa, seq=False)

        loss = None
        if labels is not None: loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=0)

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

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False, load_saved=False, save_dataset=False, cache='h:/hf', extract_args=None, max_ctx=2048):

    if load_saved:
        if cache is None:
            cache = 'h:/cache'
        else:
            cache = cache

        os.makedirs(cache, exist_ok=True)
        train = os.path.join(cache, "train.arrow")
        test = os.path.join(cache, "test.arrow")
        if os.path.exists(train) and os.path.exists(test):
            from datasets import Dataset
            train = Dataset.load_from_disk(train)
            test = Dataset.load_from_disk(test)
            return train, test   

    def filter(x):
        return (0 < len(x["sentence"]) and 0 < len(x["audio"]["array"])) 

    base_dir = "H:/gf"

    train = load_dataset("fixie-ai/common_voice_17_0", "en", split="train", streaming=True, token=token, trust_remote_code=True).take(10000)
    test = load_dataset("fixie-ai/common_voice_17_0", "en", split="test", streaming=True, token=token, trust_remote_code=True).take(1000)

    train = train.filter(filter).cast_column("audio", Audio(sampling_rate=sample_rate))
    test = test.filter(filter).cast_column("audio", Audio(sampling_rate=sample_rate))

    train = train.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "sentence"])
    test = test.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "sentence"])
    train.save_disk(train) if save_dataset is True else None
    test.save_disk(test) if save_dataset is True else None
    return train, test

def main():
    token = ""
    log_dir = os.path.join('./ignore/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("./tokenizer.json") 
    
    extract_args = {

        "spectrogram": True,
        "pitch": True,
        "waveform": False,
        "harmonics": False,
        "aperiodics": False,
        "phase": False,
        "hilbert": False,
        "pitch_tokens": True,
        "hop_length": 160,
        "sample_rate": 16000
    }

    param = Dimensions(tokens=40000, mels=128, dims=512, head=4, layer=8, act="gelu", n_type="layernorm")

    train_dataset, test_dataset = prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache=None, extract_args=extract_args, max_ctx=2048)

    model = Model(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    metrics_fn = partial(compute_metrics, print_pred=True, num_samples=1, tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=10000,
        eval_steps=1000,
        warmup_steps=100,
        logging_steps=100,
        logging_dir=log_dir,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=5000,
        report_to=["tensorboard"],
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        eval_on_start=False,
        batch_eval_metrics=False,
        disable_tqdm=False,
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        learning_rate=0.00025,
        weight_decay=0.025,
        dataloader_pin_memory=False,
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

    trainer.train()

if __name__ == "__main__":
    main()

