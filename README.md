Standard asr model with RoPE, optional blending of scaled dot product and cosine similarity, and optional blending of spectrogram and waveform input. 
Both can be variable controlled or turned off. Full script with tranining loop compatable with hugging face. For testing.

set input_features=True, waveform=True:

            def prepare_dataset(batch, input_features=True, waveform=False):
                audio = batch["audio"]
                if input_features:
                    batch["input_features"] = extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
                if waveform:
                    batch["waveform"] = torch.tensor(audio["array"]).unsqueeze(0).float()
                batch["labels"] = tokenizer(batch["transcription"]).input_ids
                return batch

      ### control blending amount: 
      
      class AudioEncoder(nn.Module):
          def __init__(self, mels: int, ctx: int, dims: int, head: int, layer, act: str = "relu"):
              super().__init__()
      
              self.dropout = 0.1
              act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), 
                         "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
              self.act = act_map.get(act, nn.GELU())
              self.blend = nn.Parameter(torch.tensor(0.5)) # Learnable blend factor between 0 and 1 
      
              self.se = nn.Sequential(Conv1d(mels, dims, kernel_size=3, padding=1), self.act, 
                  Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2), 
                  Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),     
                  Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
                  nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1))
              
              self.we = nn.Sequential(
                  nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
                  nn.GELU(),
                  nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
                  nn.GELU(),
                  nn.AdaptiveAvgPool1d(ctx),
              )
      
              self.register_buffer("positional_embedding", sinusoids(ctx, dims))       
      
              self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, cross_attention=False, act = "relu")
                          for _ in range(layer)]) if layer > 0 else None)
              
              self.ln_enc = RMSNorm(normalized_shape=dims)
      
          def forward(self, x, w) -> Tensor:
              """ x : torch.Tensor, shape = (batch, mels, ctx) the mel spectrogram of the audio """
              """ w : torch.Tensor, shape = (batch, 1, ctx) the waveform of the audio """
      
              if x is not None:
                  if w is not None:
                      x = self.se(x) 
                      x = x.permute(0, 2, 1) 
                      assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                      x = (x + self.positional_embedding).to(x.dtype)
                      w = self.we(w).permute(0, 2, 1)
                      blend = torch.sigmoid(self.blend)
                      x = blend * x + (1 - blend) * w
                  else:
                      x = self.se(x) 
                      x = x.permute(0, 2, 1) 
                      assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                      x = (x + self.positional_embedding).to(x.dtype)
                  
              else:
                  assert w is not None, "You have to provide either x or w"
                  x = self.we(w).permute(0, 2, 1)
                  assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                  x = (x + self.positional_embedding).to(x.dtype)
      
              for block in chain(self.blockA or []):
                  x = block(x)
              return self.ln_enc(x)

    
    class Multihead(nn.Module):
        blend = True
        mag = False
        def __init__(self, dims: int, head: int):
            super().__init__()
            self.dims = dims
            self.head = head
            head_dim = dims // head
            self.head_dim = head_dim
            self.dropout = 0.1
    
            self.q = Linear(dims, dims)
            self.k = Linear(dims, dims, bias=False)
            self.v = Linear(dims, dims)
            self.o = Linear(dims, dims)
    
            self.rotary = Rotary(dim=head_dim, learned_freq=True)
    
            if Multihead.blend:
                self.factor = nn.Parameter(torch.tensor(0.5, **tox))
    
        def compute_cosine_attention(self, q: Tensor, k: Tensor, v: Tensor, mask):
            ctx = q.shape[1]
            qn = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
            kn = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
            qk = torch.matmul(qn, kn.transpose(-1, -2))
    
            if Multihead.mag:
                qm = torch.norm(q, dim=-1, keepdim=True)
                km = torch.norm(k, dim=-1, keepdim=True)
                ms = (qm * km.transpose(-1, -2)) ** 0.5
                ms = torch.clamp(ms, min=1e-8)
                qk = qk * ms
    
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
    
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            w = F.dropout(w, p=self.dropout, training=self.training)
            out = torch.matmul(w, v)
            return out, qk
    
        def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask = None, kv_cache = None):
            q = self.q(x)
            if kv_cache is None or xa is None or self.k not in kv_cache:
                k = self.k(x if xa is None else xa)
                v = self.v(x if xa is None else xa)
            else:
                k = kv_cache[self.k]
                v = kv_cache[self.v]
            out, qk = self._forward(q, k, v, mask)
            return self.o(out), qk
    
        def _forward(self, q: Tensor, k: Tensor, v: Tensor, mask = None):
            ctx = q.shape[1]
            dims = self.dims
            scale = (dims // self.head) ** -0.25
    
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
    
            freqs_cis = self.rotary(ctx)
            q = self.rotary.apply_rotary(q, freqs_cis)
            k = self.rotary.apply_rotary(k, freqs_cis)
    
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            w = F.dropout(w, p=self.dropout, training=self.training)
            out = torch.matmul(w, v)
    
            if Multihead.blend:
                cos_w, cos_qk = self.compute_cosine_attention(q, k, v, mask)
                blend = torch.sigmoid(self.factor)
                out = blend * cos_w + (1 - blend) * out
                qk = blend * cos_qk + (1 - blend) * qk
            
            out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach() if self.training else qk
            return out, qk
