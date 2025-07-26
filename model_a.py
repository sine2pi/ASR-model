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
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from torch.nn.functional import scaled_dot_product_attention
from echoutils import *
# from focusb import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

PATH = 'E:/hf'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def reshape_to_output(self, attn_output, batch, ctx):
    return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims).contiguous()

def qkv_init(dims: int, head: int):
    head_dim = dims // head
    q = nn.Linear(dims, dims)
    k = nn.Linear(dims, dims, bias=False)
    v = nn.Linear(dims, dims)
    o = nn.Linear(dims, dims)
    lna = nn.LayerNorm(dims, bias=False)  
    lnb = nn.LayerNorm(head_dim, bias=False)
    return q, k, v, o, lna, lnb

def create_qkv(dims, head, q, k, v, x, xa=None):
    z = default(xa, x)
    head_dim = dims // head
    scale = head_dim ** -0.25
    q = q(x) * scale
    k = k(z) * scale
    v = v(z)
    batch, ctx, dims = q.shape
    def _shape(tensor):
        return tensor.view(batch, ctx, head, head_dim).transpose(1, 2).contiguous()
    return _shape(q), _shape(k), _shape(v)

def calculate_attention(q, k, v, mask=None, temperature=1.0):
    batch, head, ctx, dims = q.shape
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5
    a = scaled_dot_product_attention(scaled_q, k, v, is_causal=mask is not None and q.shape[1] > 1)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None

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
    def __init__(self, dims: int, head: int, max_iters: int = 3, threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1):
        super(attention, self).__init__()
        
        self.q,  self.k,  self.v,  self.o, self.lna, self.lnb = qkv_init(dims, head)
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.dropout = dropout
        self.max_iters = max_iters
        self.rope = rotary(dims=dims, head=head)    

        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.factor = nn.Parameter(torch.tensor(factor))
        self.attn_local = LocalAttentionModule(self.head_dim)

    def _focus(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        z = default(xa, x)
        q, k, v = create_qkv(self.dims, self.head, self.q, self.k, self.v, self.lna(x), self.lna(z))
        # q=self.lnb(q)
        # k=self.lnb(k)
        iteration = 0
        prev_attn = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = self.threshold.item()
        factor = self.factor.item()

        q_cur = q
        while iteration < self.max_iters:
            eff_span = z.shape[1]
            if eff_span == 0: 
                break

            q_iter = q_cur[:, :, :eff_span, :]
            k_iter = k[:, :, :eff_span, :]
            v_iter = v[:, :, :eff_span, :]
            q = self.attn_local.query_module(q_iter)
            k = self.attn_local.key_module(k_iter)
            v = self.attn_local.value_module(v_iter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            q = self.rope(q, q.shape[2])
            k = self.rope(k, k.shape[2]) 

            attn_iter, _ = calculate_attention(
                self.lnb(q), self.lnb(k), v, mask=iter_mask)

            out_span = self.attn_local._reshape_to_output(attn_iter)
            if out_span.dim() == 4:
                b, h, s, d = out_span.shape
                proj_span = self.attn_local.out_proj(out_span.view(-1, d)).view(b, h, s, -1)
            elif out_span.dim() == 3:
                b, s, d = out_span.shape
                if d == self.head_dim:
                    proj_span = self.attn_local.out_proj(out_span.view(-1, d)).view(b, 1, s, -1)
                elif d == self.head * self.head_dim:
                    proj_span = out_span.view(b, self.head, s, self.head_dim)
                else:
                    raise RuntimeError(f"Cannot reshape out_span of shape {out_span.shape} to [b, h, s, head_dim]")
            else:
                raise RuntimeError(f"Unexpected out_span shape: {out_span.shape}")

            iter_out = torch.zeros_like(q_cur)
            iter_out[:, :, :eff_span, :] = proj_span
            diff = torch.abs(iter_out - prev_attn).mean()
            dthresh = threshold + factor * diff
            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_attn = iter_out.clone()
            q_cur = q_cur + iter_out
            attn_out = iter_out
            iteration += 1

        output = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(output), None

    def _slide_win_local(self, x: Tensor, win_size: int, span_len: int,
                         mask: Optional[Tensor] = None) -> Tensor:
        batch, ctx, dims = x.shape
        output = torch.zeros_like(x)
        num_win = (ctx + win_size - 1) // win_size

        for i in range(num_win):
            q_start = i * win_size
            q_end = min(q_start + win_size, ctx)
            q_len = q_end - q_start
            if q_len == 0: 
                continue

            kv_start = max(0, q_end - span_len)
            kv_end = q_end
            query_win = x[:, q_start:q_end, :]
            key_win = x[:, kv_start:kv_end, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
                elif mask.dim() == 2:
                    win_mask = mask[q_start:q_end, kv_start:kv_end]

            attn_out_win, _ = self._focus(
                x=query_win,
                xa=key_win,
                mask=win_mask)
            output[:, q_start:q_end, :] = attn_out_win
        return output

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, 
                use_sliding_window: bool = False, win_size: int = 512, span_len: int = 1024) -> Tensor:
        if use_sliding_window:
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
        self.attna = attentiona(dims, head, max_iters=3)
        self.mlp = nn.Sequential(Linear(dims, dims*4), get_activation(act), Linear(dims*4, dims))

    def forward(self, x, xa = None, mask = None) -> Tensor:

        x = x + self.attnb(self.lna(x), xa=None, mask=mask)    
        if xa is not None:
            x = x + self.attna(self.lna(x), xa, mask=None, use_sliding_window=True, win_size=500, span_len=1500)  
        x = x + self.mlp(self.lna(x))
        return x
 
class processor(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, act: str = "gelu"): 
        super(processor, self).__init__()

        self.ln = nn.LayerNorm(dims, device=device, dtype=dtype)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.posin = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)

        act_fn = get_activation(act)        
        self.encoder = nn.Sequential(
            Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        self.bA = nn.ModuleList([Residual(dims=dims, head=head, act=act_fn) for _ in range(layer)])
        self.bB = nn.ModuleList([Residual(dims=dims, head=head, act=act_fn) for _ in range(layer)])

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, sequential=False) -> Tensor:    

        x = self.token(x.long()) + self.positional[:x.shape[1]]    
        xa = self.encoder(xa).permute(0, 2, 1)
        xa = xa + self.posin(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        for b in chain(self.bA or []):
            xa = b(x=xa, xa=None, mask=None)

        for b in chain(self.bB or []):
            x = b(x=x, xa=None, mask=self.mask)
            y = b(x, xa=xa, mask=None)
            if sequential:
                x = y
            else:
                a = torch.sigmoid(self.blend)
                x = a * y + (1 - a) * x 

        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.ln(x)        
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()
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
            elif isinstance(module, attention):
                self.init_counts["attention"] += 1
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

def main():
    token = ""
    log_dir = os.path.join('D:/newmodel/output/logs/', datetime.now().strftime('%m-%d_%H_%M_%S'))
    os.makedirs(log_dir, exist_ok=True)
    tokenizer = setup_tokenizer("D:/newmodel/mod5/tokenizer.json") 

    extract_args = {
        "waveform": False,
        "spec": False,
        "f0": False,
        "f0t": False,
        "pitch": True,
        "harmonics": False,
        "aperiodics": False,
        "phase_mod": False,
        "crepe": False,        
        "sample_rate": 16000,
        "hop_length": 256,
        "mode": "mean",
        "debug": False,
    }

    param = Dimensions(
        vocab=40000,
        mels=128,
        ctx=2048,
        dims=512,
        head=4,
        layer=4,
        act="swish",
        )

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
        warmup_steps=100,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-8, weight_decay=training_args.weight_decay, betas=(0.9, 0.999), 
    amsgrad=False, foreach=False, fused=False, capturable=False, differentiable=False, maximize=False)
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
    trainer.train()
if __name__ == "__main__":

    main()
