

NLP/ASR multimodal pitch aware model. Research model.

<img width="780" alt="cc5" src="https://github.com/user-attachments/assets/ce9417de-a892-4811-b151-da612f31c0fb"  />

**This plot illustrates the pattern similiarity of pitch and spectrogram. (librispeech - clean).

To highlight the relationship between pitch and rotary embeddings, the model implements three complementary pitch-based enhancements:

1. **Pitch-modulated theta:** Pitch (f0) is used to modify the theta parameter, dynamically adjusting the rotary frequency.
2. **Direct similarity bias:** A pitch-based similarity bias is added directly to the attention mechanism.
3. **Variable radii in torch.polar:** The unit circle radius (1.0) in the `torch.polar` calculation is replaced with variable radii derived from f0. This creates acoustically-weighted positional encodings, so each position in the embedding space reflects the acoustic prominence in the original speech. This approach effectively adds phase information without significant computational overhead.

The function `torch.polar` constructs a complex tensor from polar coordinates:

````python
# torch.polar(magnitude, angle) returns:
result = magnitude * (torch.cos(angle) + 1j * torch.sin(angle))
````

So, for each element:
- **magnitude** is the modulus (radius, r)
- **angle** is the phase (theta, in radians)
- The result is: `r * exp(i * theta) = r * (cos(theta) + i * sin(theta))`

Reference: [PyTorch Documentation - torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html)

Here are the abbreviated steps for replacing theta and radius in the rotary forward:

```python
f0 = f0.to(device, dtype) # feature extracted during processing
f0_mean = f0.mean() # mean only used as theta in freqs calculation
theta = f0_mean + self.theta
freqs = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
freqs = t[:, None] * freqs[None, :]

radius = f0.to(device, dtype) # we want to avoid using the mean of f0 (or any stat or interpolation)
if radius.shape[0] != x.shape[0]: # encoder outputs will already be the correct length
    F = radius.shape[0] / x.shape[0]
    idx = torch.arange(x.shape[0], device=f0.device)
    idx = (idx * F).long().clamp(0, radius.shape[0] - 1)
    radius = radius[idx] # it's the best method i know of that retains f0 character 
radius = radius.unsqueeze(-1).expand(-1, freqs.shape[-1])
radius = torch.sigmoid(radius)
freqs = torch.polar(radius, freqs)
```

<img width="780" alt="cc4" src="https://github.com/user-attachments/assets/165a3f18-659a-4e2e-a154-a3456b667bae"  />

Each figure shows 4 subplots (one for each of the first 4 dimensions of your embeddings in the test run). These visualizations show how pitch information modifies position encoding patterns in the model.

In each subplot:

- **Thick solid lines**: Standard RoPE rotations for even dimensions (no F0 adaptation)
- **Thick dashed lines**: Standard RoPE rotations for odd dimensions (no F0 adaptation)
- **Thin solid lines**: F0-adapted RoPE rotations for even dimensions
- **Thin dashed lines**: F0-adapted RoPE rotations for odd dimensions


1. **Differences between thick and thin lines**: This shows how much the F0 information is modifying the standard position encodings. Larger differences indicate stronger F0 adaptation.

2. **Pattern changes**: The standard RoPE (thick lines) show regular sinusoidal patterns, while the F0-adapted RoPE (thin lines) show variations that correspond to the audio's pitch contour.

3. **Dimension-specific effects**: Compared across four subplots to see if F0 affects different dimensions differently.

4. **Position-specific variations**: In standard RoPE, frequency decreases with dimension index, but F0 adaptation modify this pattern.

The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

<img width="680" alt="cc" src="https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8"  />
<img width="680" alt="cc2" src="https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1"  />

----


#### Diagnostic test run where 1 epoch = 1000 steps = 1000 samples:

<img width="680" alt="1555" src="https://github.com/user-attachments/assets/5bed0421-e32f-4234-ab55-51d64eb927ef" />

<img width="680" alt="1555" src="https://github.com/user-attachments/assets/14276b99-cf96-4022-9a16-4ac8ed1f6404" />

----

<img width="283" alt="f0" src="https://github.com/user-attachments/assets/00dba6f6-a943-4361-8160-5651b332034c" />

<img width="84" alt="4" src="https://github.com/user-attachments/assets/6d2c640a-3e01-4632-9cc2-7ced3249f8c5" />


------

<div>
      
</div>
<table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Wer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>45.405200</td>
      <td>42.906246</td>
      <td>86.916342</td>
    </tr>
    <tr>
      <td>200</td>
      <td>28.778900</td>
      <td>30.122574</td>
      <td>93.531128</td>
    </tr>
    <tr>
      <td>300</td>
      <td>22.751900</td>
      <td>20.838696</td>
      <td>67.996109</td>
    </tr>
    <tr>
      <td>400</td>
      <td>13.976900</td>
      <td>12.902719</td>
      <td>53.988327</td>
    </tr>
    <tr>
      <td>500</td>
      <td>4.846700</td>
      <td>6.163748</td>
      <td>32.684825</td>
    </tr>
    <tr>
      <td>600</td>
      <td>2.946100</td>
      <td>3.432577</td>
      <td>18.774319</td>
    </tr>
    <tr>
      <td>700</td>
      <td>1.530200</td>
      <td>2.647515</td>
      <td>18.822957</td>
    </tr>
    <tr>
      <td>800</td>
      <td>1.228600</td>
      <td>1.896243</td>
      <td>15.710117</td>
    </tr>
    <tr>
      <td>900</td>
      <td>0.733700</td>
      <td>1.441605</td>
      <td>11.867704</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>1.057500</td>
      <td>0.874806</td>
      <td>8.998054</td>
    </tr>
    <tr>
      <td>1100</td>
      <td>0.252500</td>
      <td>1.118475</td>
      <td>10.894942</td>
    </tr>
    <tr>
      <td>1200</td>
      <td>0.334800</td>
      <td>1.420100</td>
      <td>11.429961</td>
    </tr>
    <tr>
      <td>1300</td>
      <td>0.302600</td>
      <td>1.224957</td>
      <td>9.192607</td>
    </tr>
    <tr>
      <td>1400</td>
      <td>0.618900</td>
      <td>1.096631</td>
      <td>8.657588</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.964900</td>
      <td>0.775765</td>
      <td>7.879377</td>
    </tr>
    <tr>
      <td>1600</td>
      <td>0.225400</td>
      <td>0.666205</td>
      <td>6.322957</td>
    </tr>
    <tr>
      <td>1700</td>
      <td>0.584200</td>
      <td>0.577828</td>
      <td>6.517510</td>
    </tr>
    <tr>
      <td>1800</td>
      <td>0.098300</td>
      <td>0.507692</td>
      <td>5.933852</td>
    </tr>
    <tr>
      <td>1900</td>
      <td>0.032400</td>
      <td>0.542363</td>
      <td>7.003891</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.399700</td>
      <td>0.477278</td>
      <td>6.128405</td>
    </tr>
    <tr>
      <td>2100</td>
      <td>0.254100</td>
      <td>0.276226</td>
      <td>3.891051</td>
    </tr>
    <tr>
      <td>2200</td>
      <td>0.179300</td>
      <td>0.356367</td>
      <td>4.912451</td>
    </tr>
    <tr>
      <td>2300</td>
      <td>0.195500</td>
      <td>0.322132</td>
      <td>3.745136</td>
    </tr>
    <tr>
      <td>2400</td>
      <td>0.014300</td>
      <td>0.400927</td>
      <td>5.301556</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>0.062400</td>
      <td>0.378821</td>
      <td>4.571984</td>
    </tr>
    <tr>
      <td>2600</td>
      <td>0.112600</td>
      <td>0.516686</td>
      <td>4.669261</td>
    </tr>
    <tr>
      <td>2700</td>
      <td>0.010100</td>
      <td>0.397214</td>
      <td>4.571984</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>0.216500</td>
      <td>0.382666</td>
      <td>3.842412</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>0.021600</td>
      <td>0.475567</td>
      <td>3.550584</td>
    </tr>
    <tr>
      <td>3000</td>
      <td>0.049500</td>
      <td>0.361212</td>
      <td>4.863813</td>
    </tr>
    <tr>
      <td>3100</td>
      <td>0.010900</td>
      <td>0.426143</td>
      <td>4.474708</td>
    </tr>
    <tr>
      <td>3200</td>
      <td>0.000000</td>
      <td>0.375904</td>
      <td>3.939689</td>
    </tr>
    <tr>
      <td>3300</td>
      <td>0.096500</td>
      <td>0.275490</td>
      <td>4.134241</td>
    </tr>
    <tr>
      <td>3400</td>
      <td>0.333000</td>
      <td>0.294802</td>
      <td>3.988327</td>
    </tr>
    <tr>
      <td>3500</td>
      <td>0.077300</td>
      <td>0.336880</td>
      <td>4.328794</td>
    </tr>
    <tr>
      <td>3600</td>
      <td>0.518300</td>
      <td>0.269225</td>
      <td>4.085603</td>
    </tr>
    <tr>
      <td>3700</td>
      <td>0.012500</td>
      <td>0.279636</td>
      <td>3.501946</td>
    </tr>
    <tr>
      <td>3800</td>
      <td>0.089900</td>
      <td>0.387448</td>
      <td>4.085603</td>
    </tr>
    <tr>
      <td>3900</td>
      <td>0.094100</td>
      <td>0.298909</td>
      <td>3.307393</td>
    </tr>
    <tr>
      <td>4000</td>
      <td>0.445400</td>
      <td>0.266931</td>
      <td>3.404669</td>
    </tr>
    <tr>
      <td>4100</td>
      <td>0.105300</td>
      <td>0.291892</td>
      <td>2.772374</td>
    </tr>
    <tr>
      <td>4200</td>
      <td>0.000100</td>
      <td>0.264607</td>
      <td>2.723735</td>
    </tr>
    <tr>
      <td>4300</td>
      <td>0.001700</td>
      <td>0.232140</td>
      <td>2.431907</td>
    </tr>
    <tr>
      <td>4400</td>
      <td>0.030800</td>
      <td>0.157472</td>
      <td>1.702335</td>
    </tr>
    <tr>
      <td>4500</td>
      <td>0.004200</td>
      <td>0.181895</td>
      <td>2.237354</td>
    </tr>
    <tr>
      <td>4600</td>
      <td>0.010300</td>
      <td>0.192630</td>
      <td>1.945525</td>
    </tr>
    <tr>
      <td>4700</td>
      <td>0.000000</td>
      <td>0.128976</td>
      <td>2.431907</td>
    </tr>
    <tr>
      <td>4800</td>
      <td>0.008100</td>
      <td>0.140460</td>
      <td>2.772374</td>
    </tr>
    <tr>
      <td>4900</td>
      <td>0.017000</td>
      <td>0.141039</td>
      <td>2.285992</td>
    </tr>
    <tr>
      <td>5000</td>
      <td>0.000300</td>
      <td>0.185659</td>
      <td>2.285992</td>
    </tr>
    <tr>
      <td>5100</td>
      <td>0.019600</td>
      <td>0.135166</td>
      <td>2.383268</td>
    </tr>
    <tr>
      <td>5200</td>
      <td>0.011700</td>
      <td>0.163442</td>
      <td>2.237354</td>
    </tr>
    <tr>
      <td>5300</td>
      <td>0.000100</td>
      <td>0.152560</td>
      <td>2.431907</td>
    </tr>
    <tr>
      <td>5400</td>
      <td>0.000000</td>
      <td>0.151676</td>
      <td>2.091440</td>
    </tr>
    <tr>
      <td>5500</td>
      <td>0.000000</td>
      <td>0.159987</td>
      <td>2.237354</td>
    </tr>
    <tr>
      <td>5600</td>
      <td>0.000000</td>
      <td>0.108741</td>
      <td>2.285992</td>
    </tr>
    <tr>
      <td>5700</td>
      <td>0.000000</td>
      <td>0.130140</td>
      <td>1.848249</td>
    </tr>
    <tr>
      <td>5800</td>
      <td>0.000000</td>
      <td>0.120961</td>
      <td>1.994163</td>
    </tr>
    <tr>
      <td>5900</td>
      <td>0.000000</td>
      <td>0.125711</td>
      <td>1.556420</td>
    </tr>
    <tr>
      <td>6000</td>
      <td>0.000000</td>
      <td>0.160521</td>
      <td>1.848249</td>
    </tr>
    <tr>
      <td>6100</td>
      <td>0.000000</td>
      <td>0.119946</td>
      <td>1.799611</td>
    </tr>
    <tr>
      <td>6200</td>
      <td>0.030100</td>
      <td>0.134362</td>
      <td>1.507782</td>
    </tr>
    <tr>
      <td>6300</td>
      <td>0.000000</td>
      <td>0.210862</td>
      <td>1.848249</td>
    </tr>
    <tr>
      <td>6400</td>
      <td>0.000000</td>
      <td>0.185011</td>
      <td>1.605058</td>
    </tr>
    <tr>
      <td>6500</td>
      <td>0.163300</td>
      <td>0.210803</td>
      <td>1.605058</td>
    </tr>
    <tr>
      <td>6600</td>
      <td>0.006900</td>
      <td>0.198053</td>
      <td>1.410506</td>
    </tr>
    <tr>
      <td>6700</td>
      <td>0.000000</td>
      <td>0.212066</td>
      <td>1.410506</td>
    </tr>
    <tr>
      <td>6800</td>
      <td>0.172800</td>
      <td>0.193078</td>
      <td>1.507782</td>
    </tr>
    <tr>
      <td>6900</td>
      <td>0.000000</td>
      <td>0.172742</td>
      <td>1.410506</td>
    </tr>

  </tbody>
</table><p>

