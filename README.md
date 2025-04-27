ASR encoder-decoder model with optional blending of spectrogram and waveform input. Full script with tranining loop compatable with hugging face. For testing.

This model's learnable blend (with a sigmoid-mixed parameter) between waveform and spectrogram encodings is a novel and practical way to let the model decide the optimal mix. This form of adaptive fusion is less common in open-source ASR codebases.

Blending waveform and spectrogram features has been explored in some research, but is not standard in ASR pipelines.
This learnable blend is a modern, under-explored approach addressing the waveform spectrogram debate. Initial findings of the pilot run suggest that the blending of the two significantly decreases WER compared to standalone waveform and spectrogram without significantly increasing overhead. 

This model uses 0 for padding masking and silence and as such the attention mechanism uses multiplicative masking instead of additive. The 0.001 is so that the model can still learn to identify silence. This gives silence tokens a tiny but meaningful attention weight rather than completely masking them out. This is conceptually sound because:

Silence/pauses in speech carry rhythmic and semantic information.
The 0.001 factor means silence is "whispered" to the model rather than "shouted".
The model can learn timing patterns where pauses are meaningful.



    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        freq = self.rotary(ctx)
        q = self.rotary.apply_rotary(q, freq)
        k = self.rotary.apply_rotary(k, freq)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        scaling_factors_causal = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0))

        token_ids = k[:, :, :, 0]
        scaling_factors_silence = torch.ones_like(token_ids).to(q.device, q.dtype)
        scaling_factors_silence[token_ids == 0] = 0.001
        scaling_factors_silence = scaling_factors_silence.unsqueeze(-2).expand(qk.shape).to(q.device, q.dtype)

        combined_scaling_factors = scaling_factors_causal.unsqueeze(0).to(q.device, q.dtype)  * scaling_factors_silence.to(q.device, q.dtype)
        qk *= combined_scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, qk.detach()

3 tests: spectrogram, waveform, spectrogram+waveform.
### IN PROGRESS

