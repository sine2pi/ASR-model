import os
import math
import warnings
import logging
from itertools import chain
import torch
import torch.nn.functional as feature
from torch import nn, Tensor
from typing import Optional, Dict, Union, List, Tuple
import numpy as np
from functools import partial
from datetime import datetime
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from echoutils import *

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
    def forward(self, x, ctx) -> Tensor:
        freqs = (self.theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        t = torch.arange(ctx, device=device, dtype=dtype) 
        freqs = t[:, None] * freqs
        freqs=torch.polar(torch.ones_like(freqs), freqs)
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

class attention(nn.Module):
    def __init__(self, dims: int, head: int):
        super(attention, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.q = nn.Linear(dims, dims).to(device, dtype)
        self.k = nn.Linear(dims, dims, bias=False).to(device, dtype)
        self.v = nn.Linear(dims, dims).to(device, dtype)
        self.o = nn.Linear(dims, dims).to(device, dtype)
        self.rope = rotary(dims=dims, head=head)
        self.lny = nn.LayerNorm(self.head_dim, bias = False)  
        self.lnx = nn.LayerNorm(dims, bias = False)
    def forward(self, x: Tensor, xa = None, mask = None):
        scale = (self.dims // self.head) ** -0.25
        q = self.q(self.lnx(x))
        k = self.k(self.lnx(x if xa is None else xa))
        v = self.v(self.lnx(x if xa is None else xa))
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        q = self.rope(q, q.shape[2])
        k = self.rope(k, k.shape[2]) 
        a = scaled_dot_product_attention(self.lny(q), self.lny(k), v, is_causal=mask is not None and q.shape[1] > 1)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.o(out)

class tgate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gate_projections = nn.ModuleList([
            nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            for _ in range(num_types)])
        self.type_classifier = nn.Sequential(
            Linear(dims, num_types),
            nn.Softmax(dim=-1))
    def forward(self, x):
        type_probs = self.type_classifier(x)
        gates = torch.stack([gate(x) for gate in self.gate_projections], dim=-1)
        comb_gate = torch.sum(gates * type_probs.unsqueeze(2), dim=-1)
        return comb_gate

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, dims: int, head: int, act: str = "silu"):
        super().__init__()
        act_fn = get_activation(act)
        self.blend = nn.Parameter(torch.tensor(0.5)) 
        self.attn = attention(dims, head)
        self.mlp = nn.Sequential(Linear(dims, dims*4), act_fn, Linear(dims*4, dims))
        self.tgate = tgate(dims=dims, num_types=4*2)
        self.lna = nn.LayerNorm(dims, bias = False)
    def forward(self, x, xa=None, mask=None) -> Tensor:
        xb = x + self.attn(self.lna(x), xa=None, mask=mask)[0]
        if xa is not None:
            x = x + self.attn(self.lna(x), xa=xa, mask=None)[0] 
            b = torch.sigmoid(self.blend)
            x = b * xb + (1 - b) * x   
        out = self.mlp(self.lna(x))
        gate = self.tgate(self.lna(x)) 
        x = x + gate * out
        return x

class processor(nn.Module):
    def __init__(self, vocab: int, mels: int, ctx: int, dims: int, head: int, layer: int, act: str = "gelu"): 
        super(processor, self).__init__()
        act_fn = get_activation(act)
        self.token = nn.Embedding(vocab, dims, device=device, dtype=dtype)
        self.positional = nn.Parameter(torch.empty(ctx, dims, device=device, dtype=dtype), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
        self.positional_sin = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)
        self.encoder = nn.Sequential(
            Conv1d(1, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)
        self.bA = nn.ModuleList([Residual(dims=dims, head=head, act=act_fn) for _ in range(layer)])
        self.bB = nn.ModuleList([Residual(dims=dims, head=head, act=act_fn) for _ in range(layer)])
        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.norm = nn.LayerNorm(dims, device=device, dtype=dtype)

    def forward(self, x, xa) -> Tensor:    
        x = self.token(x.long()) + self.positional[:x.shape[1]]
        xa = self.encoder(xa).permute(0, 2, 1)
        xa = xa + self.positional_sin(xa.shape[1], xa.shape[-1], 10000.0).to(device, dtype)
        for b in chain(self.bA or []):
            xa = b(x=xa, xa=None, mask=None)
        for b in chain(self.bB or []):
            x = b(x=x, xa=None, mask=self.mask)
            x = b(x, xa=xa, mask=None)
        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.norm(x)
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
            act=param.act,
            )       
        
    def forward(self,
        labels=None, input_ids=None, pitch: Optional[torch.Tensor]=None) -> Dict[str, Optional[torch.Tensor]]:
        if pitch is not None:
            xa = pitch
        x = input_ids
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
    log_dir = os.path.join('D:/newmodel/output/logs', datetime.now().strftime('%m-%d_%H_%M_%S'))
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
        load_saved=False, save_dataset=False, cache_dir=None, extract_args=None, max_ctx=param.ctx)

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
