

NLP/ASR multimodal modal with f0-modulated relative positional embeddings. For research/testing.

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
      <td>43.799500</td>
      <td>41.256195</td>
      <td>93.628405</td>
    </tr>
    <tr>
      <td>200</td>
      <td>30.375300</td>
      <td>30.783518</td>
      <td>96.692607</td>
    </tr>
    <tr>
      <td>300</td>
      <td>22.879200</td>
      <td>23.398602</td>
      <td>79.717899</td>
    </tr>
    <tr>
      <td>400</td>
      <td>17.802800</td>
      <td>16.250486</td>
      <td>64.396887</td>
    </tr>
    <tr>
      <td>500</td>
      <td>11.074700</td>
      <td>10.244286</td>
      <td>46.692607</td>
    </tr>
    <tr>
      <td>600</td>
      <td>7.080400</td>
      <td>5.505992</td>
      <td>28.745136</td>
    </tr>
    <tr>
      <td>700</td>
      <td>3.024100</td>
      <td>3.408702</td>
      <td>20.476654</td>
    </tr>
    <tr>
      <td>800</td>
      <td>2.081900</td>
      <td>2.144920</td>
      <td>16.245136</td>
    </tr>
    <tr>
      <td>900</td>
      <td>1.798200</td>
      <td>1.900899</td>
      <td>10.554475</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>1.536500</td>
      <td>1.017085</td>
      <td>9.970817</td>
    </tr>
    <tr>
      <td>1100</td>
      <td>0.499300</td>
      <td>0.987791</td>
      <td>8.657588</td>
    </tr>
    <tr>
      <td>1200</td>
      <td>0.919400</td>
      <td>1.139080</td>
      <td>9.581712</td>
    </tr>
    <tr>
      <td>1300</td>
      <td>0.396500</td>
      <td>1.316054</td>
      <td>12.013619</td>
    </tr>
    <tr>
      <td>1400</td>
      <td>0.650500</td>
      <td>1.166937</td>
      <td>7.879377</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.509600</td>
      <td>1.039125</td>
      <td>8.852140</td>
    </tr>
    <tr>
      <td>1600</td>
      <td>0.331300</td>
      <td>0.602630</td>
      <td>6.371595</td>
    </tr>
    <tr>
      <td>1700</td>
      <td>0.710900</td>
      <td>0.587328</td>
      <td>6.468872</td>
    </tr>
    <tr>
      <td>1800</td>
      <td>0.250300</td>
      <td>0.450811</td>
      <td>4.766537</td>
    </tr>
    <tr>
      <td>1900</td>
      <td>0.323300</td>
      <td>0.306493</td>
      <td>3.599222</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.034700</td>
      <td>0.438065</td>
      <td>4.280156</td>
    </tr>
    <tr>
      <td>2100</td>
      <td>0.433600</td>
      <td>0.401657</td>
      <td>5.058366</td>
    </tr>
    <tr>
      <td>2200</td>
      <td>0.123000</td>
      <td>0.382323</td>
      <td>4.620623</td>
    </tr>
    <tr>
      <td>2300</td>
      <td>0.353100</td>
      <td>0.316831</td>
      <td>3.891051</td>
    </tr>
    <tr>
      <td>2400</td>
      <td>0.000300</td>
      <td>0.207764</td>
      <td>3.307393</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>0.145000</td>
      <td>0.437847</td>
      <td>4.571984</td>
    </tr>
    <tr>
      <td>2600</td>
      <td>0.023600</td>
      <td>0.214705</td>
      <td>2.626459</td>
    </tr>
    <tr>
      <td>2700</td>
      <td>0.093100</td>
      <td>0.239788</td>
      <td>2.918288</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>0.169300</td>
      <td>0.258873</td>
      <td>2.626459</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>0.029800</td>
      <td>0.228420</td>
      <td>2.577821</td>
    </tr>
    <tr>
      <td>3000</td>
      <td>0.072300</td>
      <td>0.218732</td>
      <td>2.529183</td>
    </tr>
    <tr>
      <td>3100</td>
      <td>0.114100</td>
      <td>0.297831</td>
      <td>2.626459</td>
    </tr>
    <tr>
      <td>3200</td>
      <td>0.162000</td>
      <td>0.250610</td>
      <td>2.431907</td>
    </tr>
    <tr>
      <td>3300</td>
      <td>0.032100</td>
      <td>0.247746</td>
      <td>2.529183</td>
    </tr>
    <tr>
      <td>3400</td>
      <td>0.000500</td>
      <td>0.220101</td>
      <td>2.529183</td>
    </tr>
    <tr>
      <td>3500</td>
      <td>0.155700</td>
      <td>0.462472</td>
      <td>3.599222</td>
    </tr>
    <tr>
      <td>3600</td>
      <td>0.061200</td>
      <td>0.243225</td>
      <td>2.042802</td>
    </tr>
    <tr>
      <td>3700</td>
      <td>0.059000</td>
      <td>0.157827</td>
      <td>1.896887</td>
    </tr>
    <tr>
      <td>3800</td>
      <td>0.081300</td>
      <td>0.142783</td>
      <td>2.188716</td>
    </tr>
    <tr>
      <td>3900</td>
      <td>0.000000</td>
      <td>0.152411</td>
      <td>1.945525</td>
    </tr>
    <tr>
      <td>4000</td>
      <td>0.148700</td>
      <td>0.162399</td>
      <td>1.264591</td>
    </tr>
    
  </tbody>
</table><p>

