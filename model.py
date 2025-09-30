import os, torch
from datasets import load_dataset, Audio
from torch.nn.functional import scaled_dot_product_attention
from torch import nn, Tensor
from typing import Iterable
import numpy as np
from functools import partial
from datetime import datetime
from dataclasses import dataclass
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from einops.layers.torch import Rearrange
from echoutils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

THETA = 30000.0

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
    num_features: int = 0

def sinusoids(ctx, dims, theta=THETA):
    tscales = torch.exp(-torch.log(torch.tensor(float(theta))) / (dims // 2 - 1) * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    positional_embedding = nn.Parameter(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) , requires_grad=True)
    return positional_embedding    

class AudioEncoder(nn.Module):
    def __init__(n, mels, dims, head, act, norm_type, norm=False, enc=False):
        super().__init__()
        
        n.act_fn = get_activation(act)
        n.conv1 = nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1)
        n.conv2 = nn.Conv1d(1, dims, kernel_size=3, stride=1, padding=1)
        
        n.encoder = nn.Sequential(
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), 
            n.act_fn, nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), n.act_fn)   

        theta = nn.Parameter(torch.tensor(THETA), requires_grad=True)
        n.audio = lambda length, dims: sinusoids(length, dims, theta)
        n.EncoderLayer = nn.TransformerEncoderLayer(d_model=dims, nhead=head, batch_first=True) if enc else nn.Identity()
        n.ln = get_norm(norm_type, dims) if norm else nn.Identity()
               
    def forward(n, x):
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

class rotary(nn.Module):
    def __init__(n, dims, head):
        super().__init__()

        n.head_dim = dims // head
        n._head = nn.Linear(dims, 1) 

    def _compute_freqs(n, mask=None):
        if mask is None:
            scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
            return 200 * scale / 1000
        else:
            return torch.arange(0, n.head_dim, 2, device=device, dtype=dtype) / n.head_dim * torch.log(torch.tensor(THETA))

    def radius(n, xa, freqs, mask=None):
        if mask is None:
            per_step = n._head(xa).squeeze(-1)  
            radius = torch.clamp(per_step / 100.0, 0.5, 2.0)
            return torch.polar(radius.unsqueeze(-1), freqs.unsqueeze(0))
        else:
            return torch.polar(torch.ones_like(freqs).unsqueeze(0), freqs.unsqueeze(0))

    def forward(n, x, xa=None, mask=None):

        ctx = x.shape[2]
        t = torch.arange(ctx, device=device, dtype=dtype).float()
        freqs = torch.einsum('i,j->ij', t,  n._compute_freqs(mask))
        freqs = n.radius(xa, freqs, mask=mask)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        rotary_out = torch.cat([x1.type_as(x), x2], dim=-1)
        return rotary_out
    
class attention(nn.Module):
    def __init__(n, dims, head, norm_type, modal=False): 
        super().__init__()

        n.scale = (dims // head) ** -0.25
        n.modal = modal

        n.q   = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims), Rearrange('b c (h d) -> b h c d', h = head))
        n.kv  = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims * 2), Rearrange('b c (kv h d) -> kv b h c d', kv = 2, h = head))
        n.out = nn.Sequential(Rearrange('b h c d -> b c (h d)'), nn.Linear(dims, dims))

        n.conv = nn.Conv2d(head, head, 1, bias=False) if modal else nn.Identity()
        n.ln = get_norm(norm_type, dims // head)
        n.rot = rotary(dims, head)
 
    def forward(n, x, xa=None, mask=None): 

        k, v = n.kv(x if xa is None else xa)
        q = n.q(x)
        q, k = n.rot(q, x, mask=None), n.rot(k, x if xa is None else xa, mask=None)
        a = scaled_dot_product_attention(n.ln(q), n.ln(k), v, is_causal=mask is not None)

        if n.modal and xa is not None:
            (ka, va), (kb, vb) = n.kv(x), n.kv(xa)
            qa, qb, ka, kb = n.rot(qa), n.rot(qb), n.rot(ka), n.rot(kb)
            b = scaled_dot_product_attention(n.ln(qa), n.ln(kb), vb, is_causal=mask is not None)
            c = scaled_dot_product_attention(n.ln(qb), n.ln(ka), va, is_causal=mask is not None)
            return n.out(a), n.out(n.conv(b)), n.out(n.conv(c))
        else:
            return n.out(a)
    
class gate(nn.Module):
    def __init__(n, dims, num_features, expand):
        super().__init__()
        num_features = num_features + 2 
        n.expand = min(expand, max(1, num_features))
        n.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()) for _ in range(num_features)])
        n.features = nn.Sequential(nn.Linear(dims, num_features), nn.Softmax(dim=-1))
        n.top_features = nn.Linear(dims, num_features)
        n.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(n, x):
        soft_types = n.features(x)
        types = n.top_features(x)
        n_types, n_indices = torch.topk(types, n.expand, dim=-1)
        values = F.softmax(n_types, dim=-1)
        s_types = torch.zeros_like(soft_types)
        s_types.scatter_(-1, n_indices, values)
        gates = torch.stack([gate(x) for gate in n.gates], dim=-1)
        features = torch.sigmoid(n.alpha) * s_types + (1 - torch.sigmoid(n.alpha)) * soft_types
        return torch.sum(gates * features.unsqueeze(2), dim=-1)

class residual(nn.Module):
    def __init__(n, dims, head, layer, act, norm_type, num_features):
        super().__init__()

        expand = max(1, min(4, num_features)) 
        n.ln = get_norm(norm_type, dims)
        n.act_fn = get_activation(act)

        n.attn = attention(dims, head, norm_type)
        n.gate = gate(dims, num_features, expand) if num_features > 0 else nn.Identity()
        n.mlp = nn.Sequential(get_norm(norm_type, dims), nn.Linear(dims, dims*expand), get_activation(act), nn.Linear(dims*expand, dims))

    def forward(n, x, xa=None, mask=None):

        x = x + n.attn(n.ln(x), mask=mask) 
        if xa is not None:
            xa = xa + n.gate(xa)
            x = x + n.attn(n.ln(x), xa)
        return x + n.mlp(x)

class processor(nn.Module):
    def __init__(n, tokens, mels, ctx, dims, head, layer, act, norm_type, num_features): 
        super().__init__()

        n.ln = get_norm(norm_type, dims)
        n.token = nn.Embedding(tokens, dims) 
        n.positions = nn.Parameter(torch.ones(ctx, dims), requires_grad=True)

        n.blend = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        n.dummy = torch.ones(1, ctx, dims, device=device, dtype=dtype)

        n.block: Iterable[residual] = nn.ModuleList(
            [residual(dims=dims, head=head, layer=layer, act=act, norm_type=norm_type, num_features=num_features) for _ in range(layer)]) 
        
        n.register_buffer("mask", torch.empty(ctx, ctx).fill_(-np.inf).triu_(1), persistent=False)

    def forward(n, x, xa=None, xb=None, xc=None, xd=None, xe=None, xf=None, xg=None, seq=False) -> Tensor:

        x = (n.token(x) + n.positions[:x.shape[-1]]).to(device, dtype)

        for i in n.block:
            x = i(x, mask=n.mask[:x.shape[1], :x.shape[1]])
            x = i(x, xa=i(xa))
            x = i(x, xa=i(aorb(xb, xa)))
            x = i(x, xa=i(aorb(xc, xa)))

            inputs = [(xa, xb),  (xb, xa), (xc, xa)]

            for a, b in inputs:
                xa = i(a if a is not None else b)
                x  = i(x, xa=xa)

            xc = torch.cat([x, sum(f for f in [xa, xb, xc] if f is not None)], dim=1)
            xm = i(x=xc[:, :x.shape[1]], xa=xc[:, x.shape[1]:])
            x = xm if seq else torch.sigmoid(n.blend) * x + (1 - torch.sigmoid(n.blend)) * xm

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
            num_features=param.num_features,
            )

        n.encoder = AudioEncoder(param.mels, param.dims, param.head, param.act, param.norm_type, norm=False, enc=False)

        n.layer = 0
        for name, module in n.named_modules():
            if name == '':
                continue
            n.layer += 1        

    def forward(n, labels=None, input_ids=None, spectrogram=None, waveform=None, pitch=None, pitch_tokens=None, 
            harmonics=None, aperiodics=None, phase=None, hilbert=None):
     
        features = {
            'xa': spectrogram,
            'xb': pitch,
            'xc': waveform,
            'xd': harmonics,
            'xe': aperiodics,
            'xf': phase,
            'xg': hilbert
        }
        
        encoded_features = {name: n.encoder(feature) if feature is not None else None for name, feature in features.items()}
        xa, xb, xc, xd, xe, xf, xg = [encoded_features[f'x{c}'] for c in 'abcdefg']
        output = n.processor(input_ids, xa, xb, xc, xd, xe, xf, xg)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.shape[-1]), 
                                                    labels.view(-1), ignore_index=0)
        return {"logits": output, "loss": loss}

    def _init_weights(n, m):
        n.counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0, "Conv2d": 0, "processor": 0, "attention": 0, "Residual": 0}
        for name, m in n.named_modules():
            if isinstance(m, nn.RMSNorm):
                n.counts["RMSNorm"] += 1
            if isinstance(m, nn.LayerNorm):
                n.counts["LayerNorm"] += 1                
            elif isinstance(m, nn.Linear):
                n.counts["Linear"] += 1

    def init_weights(n):
        print("Initializing model weights...")
        n.apply(n._init_weights)
        print("Initialization summary:")
        for module_type, count in n.counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

def prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False, load_saved=False, save_dataset=False, cache_dir='./cache', extract_args=None, max_ctx=2048):

    if load_saved:
        if cache_dir is None:
            cache_dir = './cache'
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

    def filter(x):
        return (0 < len(x["transcription"]) < max_ctx and 0 < len(x["audio"]["array"]) < max_ctx  * 160)

    train = load_dataset("google/fleurs", "en_us", token=token, split="train", trust_remote_code=True, streaming=streaming).take(1000)
    test  = load_dataset("google/fleurs", "en_us", token=token, split="test", trust_remote_code=True, streaming=streaming).take(100)

    train = train.filter(filter).cast_column("audio", Audio(sampling_rate=sample_rate))
    test = test.filter(filter).cast_column("audio", Audio(sampling_rate=sample_rate))
    train = train.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    test = test.map(lambda x: extract_features(x, tokenizer, **extract_args)).remove_columns(["audio", "transcription"])
    train.save_disk(train) if save_dataset is True else None
    test.save_disk(test) if save_dataset is True else None
    return train, test

def main():
    token = ""
    log_dir = os.path.join('./output/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = setup_tokenizer("./tokenizer.json") 
    
    extract_args = {
        "waveform": False,
        "spectrogram": True,
        "pitch": False,
        "pitch_tokens": False,
        "harmonics": False,
        "aperiodics": False,
        "phase": False,
        "hilbert": False,
    }

    num_features = sum(extract_args.values())
    param = Dimensions(tokens=40000, mels=128, ctx=2048, dims=512, head=4, layer=4, act="gelu", norm_type="layernorm", num_features=num_features)

    train_dataset, test_dataset = prepare_datasets(tokenizer, token, sanity_check=False, sample_rate=16000, streaming=False,
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=extract_args, max_ctx=param.ctx)

    model = Model(param).to('cuda')
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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

    trainer.train()

if __name__ == "__main__":
    main()
