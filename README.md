
ASR model + pitch aware relative positional embeddings. 

<img width="1363" height="732" alt="pitch_spectrogram" src="https://github.com/user-attachments/assets/ceb65e94-7df4-41b7-aa3d-c4aa4c6c0717" />

<img width="233" height="77" alt="legend" src="https://github.com/user-attachments/assets/fad84550-a199-43b3-8471-d011a9fd6f94" />

https://huggingface.co/Sin2pi/asr-model/tensorboard

Questions:

   -How can we make attention mechanisms aware of speech-specific properties?
   
   -Can we incorporate acoustic information directly into positional encodings?
   
   -Does pitch-conditioning improve speech recognition?

---


To explore the relationship between pitch and rotary embeddings, the model implements three complementary pitch based enhancements:

1. Pitch modulated theta Pitch f0 is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. Direct similarity bias: A pitch based similarity bias is added directly to the attention mechanism.
3. Variable radii in torch.polar: The unit circle radius 1.0 in the torch.polar calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase and amplitutde information without significant computational overhead.




The function `torch.polar` constructs a complex tensor from polar coordinates:

````python
# torch.polarmagnitude, angle returns:
result = magnitude * torch.cosangle + 1j * torch.sinangle
````

So, for each element:
- magnitude is the modulus radius, r
- angle is the phase theta, in radians
- The result is: `r * expi * theta = r * costheta + i * sintheta`

Reference: [PyTorch Documentation - torch.polar]https:pytorch.orgdocsstablegeneratedtorch.polar.html

Here are the abbreviated steps for replacing theta and radius in the rotary forward:

```python

    self.theta = nn.Parameter((torch.tensor(10000, device=device, dtype=dtype)), requires_grad=True)  

# This performs significantly better than standard 

   pos = torch.arange(ctx, device=device, dtype=dtype) 
   freqs = (self.theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
   freqs = pos[:, None] * freqs

# standard
        # pos = torch.arange(ctx, dtype=torch.float32, device=device).unsqueeze(1)
        # dim = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        # freqs = pos / (self.theta ** (dim / self.head_dim))
        # dim = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)



    def _apply_radii(self, freqs, f0, ctx):
        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            return torch.polar(radius.unsqueeze(-1), freqs), radius
        else:
            return torch.polar(torch.ones_like(freqs), freqs), None


 def compute_pitch_tokens(wav, sample_rate, labels, mode="mean"):
     import pyworld as pw
     wavnp = wav.numpy().astype(np.float64)
     f0_np, t = pw.dio(wavnp, sample_rate, frame_period=hop_length / sample_rate * 1000)
     f0_np = pw.stonemask(wavnp, f0_np, t, sample_rate)
     t = torch.from_numpy(t)
     audio_duration = len(wav) / sample_rate
     T = len(labels)
     tok_dur_sec = audio_duration / T
     token_starts = torch.arange(T) * tok_dur_sec
     token_ends = token_starts + tok_dur_sec
     start_idx = torch.searchsorted(t, token_starts, side="left")
     end_idx = torch.searchsorted(t, token_ends, side="right")
     pitch_tok = torch.zeros(T, dtype=torch.float32)
     for i in range(T):
         lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i]) # type: ignore
         segment = f0_np[lo:hi]
         if mode == "mean":
             pitch_tok[i] = segment.mean()
         elif mode == "median":
             pitch_tok[i] = torch.median(segment)
         else:
             pitch_tok[i] = segment[-1]
     pitch_tok[pitch_tok < 100.0] = 0.0
     bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
     f0t_tensor = torch.cat([torch.tensor([bos_pitch]), pitch_tok])
     f0t_tensor = torch.where(f0t_tensor == 0.0, torch.zeros_like(f0t_tensor), (f0t_tensor - 71.0) / (500.0 - 71.0))
     return pitch_tokens



```python

    def accumulate_phase(self, f0, t_frame, phi0=0.0):
        omega = 2 * torch.pi * f0
        dphi = omega * t_frame
        phi = torch.cumsum(dphi, dim=0) + phi0
        phi = torch.remainder(phi, 2 * torch.pi) 
        return phi
```

A closer look at whats going on. Here is a slice of the actual radius values for one step
      
      [encoder] [Radius] torch.Size[454] 92.32 [Theta] 10092.01 [f0] torch.Size[454] [Freqs] torch.Size[454, 64] 2.17+1.17j [ctx] 454
      
     [encoder] [Radius] tensor[  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           283.7590, 260.6043, 257.3410, 261.8319, 249.1852, 257.2541, 263.8165,
           272.2421, 277.6960, 286.9628, 303.8460, 305.1561, 319.5129, 330.6942,
           362.0888, 355.8571, 352.8432, 336.9354, 313.0566, 319.9086, 303.4355,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 220.4299,
           254.7619, 239.6506, 228.8830, 227.3063, 225.6784, 225.7169, 211.7767,
           223.6572, 223.4174, 222.4496, 225.1645, 228.7840, 231.8760, 228.9148,
           230.6227,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000, 205.7097, 202.8816, 182.0329, 181.6536, 180.2186,
           177.8911, 176.0775, 171.3846, 173.9602, 170.4824, 171.5723, 172.0810,
           174.3897, 177.3261, 188.3212, 188.9799, 186.7493, 221.3487,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 304.4458,
           263.3122, 251.7635, 211.8467, 207.5651, 195.3680, 184.0717, 206.3800,
           197.8661,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000, 197.6399, 195.0042, 190.7016, 187.0234, 183.5980,
           183.6842, 185.0038, 185.5778, 187.4167, 185.5085, 183.4160,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000, 190.3175, 209.7377, 206.2731, 211.9862, 219.2756, 214.3068,
           202.6881, 192.1823, 210.3404, 235.5456, 230.7845, 234.5441, 234.9773,
           241.1199, 241.9640, 237.0773, 231.6952, 238.0375, 257.9242, 264.4094,
           265.3747, 251.0286, 245.7093,   0.0000, 274.9167, 273.4767, 271.6227,
           256.5457, 245.8942, 251.3361, 240.1572, 228.9316,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000, 202.6190, 217.7865, 212.3347, 208.2926,
           209.9206, 209.3961, 210.3909, 211.6021, 202.8511, 205.1674, 211.7455,
           217.9954,   0.0000, 264.9778, 229.7112, 200.8905, 182.4680, 179.4812,
           175.4307, 172.7844, 173.6305, 172.1901, 170.5743, 167.2979, 166.7781,
           166.7783, 170.8816, 173.0406, 176.2869, 181.9142, 212.7904, 170.4449,
           173.1710, 168.3079, 154.1663,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000, 211.3942, 202.3412, 203.6764, 198.4441,
           186.2687, 209.0010, 209.5012, 214.6487, 203.8741, 195.8432, 180.9673,
             0.0000,   0.0000,   0.0000, 197.7340, 198.9476, 204.5347, 209.5858,
           204.5406, 195.1048, 198.1545, 199.8559, 207.3548, 217.9402, 217.2366,
           216.4711, 212.4731, 217.5183, 218.0658, 208.7833,   0.0000, 243.7485,
           215.1998, 235.4733, 215.3242, 215.1489, 212.6266, 203.9319, 191.8531,
           197.2219, 202.7850,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 224.1991, 167.9602,
           190.8241, 178.5659, 175.4639, 172.6353, 173.5884, 173.2250,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 209.6374, 196.2949,
           216.4672, 236.3051, 195.2339, 241.1573,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000, 195.8783, 145.3826,   0.0000,   0.0000,   0.0000,   0.0000,
             0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],
          device='cuda:0'
          
What the Radius Values Tell Us:

1. Speech Structure is Clear
   
         Zeros: Silenceunvoiced segments no pitch
         Non-zero values: Voiced speech segments with pitch
         Pattern: 0.0000 → 283.7590 → 0.0000 → 220.4299

2. Pitch Range is Realistic
   
         Range: ~145-365 Hz
         Typical speech: 80-400 Hz for most speakers
         Model values: 145-365 Hz

3. Temporal Dynamics
   
         Clusters: Pitch values cluster together natural speech
         Transitions: Smooth changes between values
         Silence gaps: Natural pauses in speech

         Silence detection: 0.0000 = no pitch silenceunvoiced
         Pitch extraction: 283.7590 = actual f0 values
         Speech segmentation: Clear boundaries between voicedunvoiced
         
         Realistic values: 145-365 Hz is normal speech range
         Proper structure: Matches natural speech patterns
         Variable radius: Working as intended

The Complex Frequency Result:

      [Freqs] torch.Size[454, 64] 2.17+1.17j


      Magnitude: sqrt2.17² + 1.17² ≈ 2.5
      Phase: atan21.17, 2.17 ≈ 0.49 radians
      
      Variable radius: Each frame has different magnitude

      Silence frames: radius ≈ 0 → freqs ≈ 0
      Voiced frames: radius ≈ 200-300 → freqs ≈ 2-3
      
      Variable attention: Important frames get more attention

      Silence: No acoustic prominence → low radius
      Speech: High acoustic prominence → high radius
      Transitions: Natural pitch changes






----




