
import warnings
import logging
from itertools import chain
import torch
from torch import nn, Tensor, einsum
import numpy as np
from dataclasses import dataclass
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def scaled_relu(x, sequence_length):
    relu_output = torch.relu(x)
    return relu_output / sequence_length

def taylor_softmax(x, order=2):
    tapprox = 1.0
    for i in range(1, order + 1):
        factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
        tapprox += x**i / factorial_i
    return tapprox / torch.sum(tapprox, dim=-1, keepdim=True)

def there_is_a(a):
    return a is not None

def AorB(a, b):
    return a if there_is_a(a) else b

def sinusoids(ctx, dims, max_tscale=10000):
    assert dims % 2 == 0
    pos = torch.log(torch.tensor(float(max_tscale))) / (dims // 2 - 1)
    tscales = torch.exp(-pos * torch.arange(dims // 2, device=device, dtype=torch.float32))
    scaled = torch.arange(ctx, device=device, dtype=torch.float32).unsqueeze(1) * tscales.unsqueeze(0)
    position = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1) 
    positional_embedding = nn.Parameter(position, requires_grad=True)
    return positional_embedding

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
        "elu": nn.ELU()
    }
    return act_map.get(act, nn.GELU())

@dataclass
class Dimensions:
    tokens: int
    mels: int
    ctx: int
    dims: int
    head: int
    head_dim: int
    layer: int
    act: str

def vectorized_taylor_sine(x, order=5):
    original_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(1, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

def vectorized_taylor_cosine(x, order=5):
    original_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(0, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

class rotary(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.taylor_order = 10

        self.theta = nn.Parameter((torch.tensor(360000, device=device, dtype=dtype)), requires_grad=False)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> torch.Tensor:
        positions = (torch.arange(0, x.shape[2], device=x.device))
        freqs = (self.theta / 220.0) * self.freqs_base
        freqs = positions[:, None] * freqs 
        freqs_rescaled = (freqs + torch.pi) % (2 * torch.pi) - torch.pi 

        with torch.autocast(device_type="cuda", enabled=False):
            cos = vectorized_taylor_cosine(freqs_rescaled, order=self.taylor_order)
            sin = vectorized_taylor_sine(freqs_rescaled, order=self.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x)

def taylor_sine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 1:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

def taylor_cosine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 0:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

class rotarya(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.taylor_order = 5

        self.theta = nn.Parameter((torch.tensor(1600, device=device, dtype=dtype)), requires_grad=False)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> torch.Tensor:

        positions = (torch.arange(0, x.shape[2], device=x.device))
        freqs = (self.theta / 220.0) * self.freqs_base
        freqs = positions[:, None] * freqs 
        freqs = (freqs + torch.pi) % (2 * torch.pi) - torch.pi
        with torch.autocast(device_type="cuda", enabled=False):
            cos = taylor_cosine(freqs, order=self.taylor_order)
            sin = taylor_sine(freqs, order=self.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x) 

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# class rotary(nn.Module):
#     def __init__(self, dims, head):
#         super(rotary, self).__init__()
#         self.dims = dims
#         self.head = head
#         self.head_dim = dims // head

#         self.theta = nn.Parameter((torch.tensor(1600, device=device, dtype=dtype)), requires_grad=False)  
#         # self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

#     def _compute_freqs_base(self):
#         mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
#         return 200 * mel_scale / 1000 

#     def forward(self, x) -> Tensor:
#         positions = (torch.arange(0, x.shape[2], device=x.device))
#         freqs = (self.theta / 220.0) * self._compute_freqs_base()
#         freqs = positions[:, None] * freqs
        
#         with torch.autocast(device_type="cuda", enabled=False):
#             freqs = torch.polar(torch.ones_like(freqs), freqs)
#             x1 = x[..., :freqs.shape[-1]*2]
#             x2 = x[..., freqs.shape[-1]*2:]
#             orig_shape = x1.shape
#             x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
#             x1 = torch.view_as_complex(x1) * freqs
#             x1 = torch.view_as_real(x1).flatten(-2)
#             x1 = x1.view(orig_shape)
#             return torch.cat([x1.type_as(x), x2], dim=-1)

class attentiona(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head

        self.pad_token = 0
        self.zmin = 1e-6
        self.zmax = 1e-5     
        self.zero = nn.Parameter(torch.tensor(1e-4, device=device, dtype=dtype), requires_grad=False)

        self.q = nn.Linear(dims, dims) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(dims // head) 
        self.rope = rotary(dims, head) 

    def forward(self, x, xa = None, mask = None,  positions = None):
        zero = self.zero

        q = self.q(x)
        k, v = self.kv(self.lna(x if xa is None else xa)).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', self.lnb(q), self.lnb(k)) * scale 

        scale = torch.ones_like(k[:, :, :, 0])
        zero = torch.clamp(F.softplus(zero), 1e-6, 1e-5)
        scale[k[:, :, :, 0].float() == 0] = zero
   
        if there_is_a(mask):
            i, j = qk.shape[-2:]
            mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            qk = qk.masked_fill(mask,  -torch.finfo(qk.dtype).max) * scale.unsqueeze(-2).expand(qk.shape)
            qk = F.sigmoid(qk)

        qk = qk * scale.unsqueeze(-2)
        qk = taylor_softmax(qk, order=2)

        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out

class tgate(nn.Module):
    def __init__(self, dims, num_types=1):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dims, dims), nn.Sigmoid()) for _ in range(num_types)])
        self.classifier = nn.Sequential(nn.Linear(dims, num_types), nn.Softmax(dim=-1))
    def forward(self, x):
        types = self.classifier(x)
        gates = torch.stack([gate(x) for gate in self.gates], dim=-1)
        cgate = torch.sum(gates * types.unsqueeze(2), dim=-1)
        return cgate

class residual(nn.Module): 
    def __init__(self, dims: int, head: int, layer = 2, act = "silu"):
        super().__init__()

        self.lna = nn.LayerNorm(dims, bias=False)       
        self.atta = attentiona(dims, head)
        self.dsl = skip_layer(dims, head, layer=2)
   
        self.tgate = tgate(dims, num_types=1)
        self.mlp = nn.Sequential(nn.Linear(dims, dims*4), get_activation(act), nn.Linear(dims*4, dims))

    def forward(self, x: Tensor, xa = None, mask = None, positions=None):
        # log = {}
        x = x + self.atta(self.lna(x), xa=xa, mask=mask)
        x, _ =  self.dsl(self.lna(x), xa=xa, mask=mask) # _ outputs logs for jumps
        x = x + self.tgate(x)
        x = x + self.mlp(self.lna(x)) 
        # print(results['jumps'])
        # log['jumps'] = l
        return x
  
class skip_layer(nn.Module):
    def __init__(self, dims, head, layer, threshold=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer = layer

        self.threshold = threshold
        self.dims = dims
        self.head = head
        self.head_dim = dims // head

        self.attention_module = attentiona(dims, head)
        self.node_predictors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, 1),
                nn.Sigmoid()
            ) for _ in range(layer)
        ])
        
        for i in range(layer):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None
            }))
        
        self.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3))
        
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        n_mlp = dims * 4
        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(dims, n_mlp), nn.GELU(), nn.Linear(n_mlp, dims))
        self.mlp_ln =nn.LayerNorm(dims)
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())

    def _calculate_shared_attention(self, x, mask=None):
        return self.attention_module(x, xa=x, mask=None)

    def predict_node_importance(self, x, layer_idx):
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.threshold).float()
    
    def forward(self, x, xa=None, mask=None):
        batch, ctx = x.shape[:2]

        working_memory = self.working_memory.expand(batch, -1, -1)
        original_x = x
        pooled_representation = x.mean(dim=1)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        
        jump_history = []
        i = 0
        while i < self.layer:
            layer = self.layers[i]
            node_importance = self.predict_node_importance(x, i)
            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
                
            norm_x = layer['ln'](x)
            importance_mask_base = node_importance.unsqueeze(1).contiguous()
            combined_custom_mask = None
            if mask is None:
                combined_custom_mask = importance_mask_base 
            else:
                combined_custom_mask = mask.contiguous() * importance_mask_base
                
            if node_importance.mean() > 0.3:
                attn_output = self._calculate_shared_attention(norm_x, mask=combined_custom_mask.contiguous())
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                
                gate_value = layer['gate'](norm_x)
                x = x + gate_value * attn_output
                memory_gate = self.memory_gate(x)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
            
            jump_prob = policy[:, 1] if i < self.layer - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            
            if should_jump:
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                i_next = min(i + jump_length, self.layer - 1)
                skip_weight = self.jump_weights[min(jump_length-1, 2)]
                x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                i = i_next
                jump_history.append(i)
            else:
                i += 1
        
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        return x, {'jumps': jump_history}
        
class processor(nn.Module):
    def __init__(self, tokens, mels, ctx, dims, head, head_dim, layer, act): 
        super(processor, self).__init__()

        act_fn = get_activation(act)   
        self.ln = nn.LayerNorm(dims)        
        self.token = nn.Embedding(tokens, dims)
        self.audio = lambda length, dims, max_tscale: sinusoids(length, dims, max_tscale)     

        self.positions = nn.Parameter(torch.empty(ctx, dims), requires_grad=True)
        self.blend = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype), requires_grad=True)
     
        self.encoder = nn.Sequential(
            nn.Conv1d(mels, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1), act_fn,
            nn.Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims), act_fn)

        modal = False
        self.block = nn.ModuleList([residual(dims, head, layer, act_fn) for _ in range(layer)]) if modal else None

        self.res = residual(dims, head, layer, act_fn)
        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)  
        self.register_buffer("mask", mask, persistent=False)

    def init_memory(self, batch):
        return torch.zeros(batch, 1, self.dims).to(next(self.parameters()).device)
    
    def update_memory(self, x, working_memory):
        return (x + working_memory) / 2 

    def forward(self, x, xa, enc=None, sequential=False, modal=False, blend=False, kv_cache=None) -> Tensor:    

        mask = self.mask[:x.shape[1], :x.shape[1]]
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token(x.long()) + self.positions[offset : offset + x.shape[-1]])

        xa = self.encoder(xa).permute(0, 2, 1)
        xa = xa + self.audio(xa.shape[1], xa.shape[-1], 36000.0).to(device, dtype)

        xa = self.res(xa, None, None)
        x  = self.res(x, None, mask)
        x  = self.res(x, xa, None)

        if blend:
            if sequential:
                y = x
            else:
                a = torch.sigmoid(self.blend)
                x = a * x + (1 - a) * y

        if modal:
            for block in chain(self.block or []):
                xm = block(torch.cat([x, xa], dim=1), mask=mask) if modal else None    
                x  = block(xm[:, :x.shape[1]], xm[:, x.shape[1]:], mask=None) if modal else x
                if blend:
                    if sequential:
                        y = x
                    else:
                        a = torch.sigmoid(self.blend)
                        x = a * x + (1 - a) * y

        x = nn.functional.dropout(x, p=0.001, training=self.training)
        x = self.ln(x)        
        x = x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()
        return x 

class Model(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        self.processor = processor(
            tokens=param.tokens,
            mels=param.mels,
            ctx=param.ctx,
            dims=param.dims,
            head=param.head,
            head_dim=param.head_dim,
            layer=param.layer,
            act=param.act)       
        
    def forward(self, labels=None, input_ids=None, pitch=None, pitch_tokens=None, spectrogram=None, waveform=None):

        x = input_ids
        xa = AorB(pitch, spectrogram) 
 
        enc = {}
        if spectrogram is not None:
            enc["spectrogram"] = spectrogram
        if waveform is not None:
            enc["waveform"] = waveform
        if pitch is not None:
            enc["pitch"] = pitch
        if pitch_tokens is not None:
            enc["ptokens"] = pitch_tokens

        logits = self.processor(x, xa, enc)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)

        return {"logits": logits, "loss": loss} 

    def _init_weights(self, module):
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "processor": 0, "attentiona": 0, "Residual": 0}
        for name, module in self.named_modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                self.init_counts["LayerNorm"] += 1                
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, residual):
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
