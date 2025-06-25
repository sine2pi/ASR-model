
### NLP/ASR multimodal pitch aware model. 
----
![3tings](https://github.com/user-attachments/assets/4158ead2-75f4-42e2-84f8-7b39b6984026)
https://huggingface.co/Sin2pi/Echo17/tensorboard?params=scalars

Pitch-Aware Processing: Integrates F0/pitch information throughout the processing pipeline, making the model sensitive to prosodic features of speech.

To highlight the relationship between pitch and rotary embeddings echo implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)*
2. The second adds direct similarity bias to attention
3. Variable radii added in place of unit circle radius(1.0) associated with torch.polar. The frequencies (f0) are time aligned with tokens creating acoustically-weighted positional encodings where the "loudness" of each position in the embedding space reflects the acoustic prominence in the original speech.

By modulating the RoPE frequencies based on pitch (F0), we are essentially telling the model to pay attention to the acoustic features relate to sequence position in a way that's proportional to the voice characteristics.  This approach creates a more speech-aware positional representation that helps the model better understand the relationship between acoustic features and text.

![rotpatterns](https://github.com/user-attachments/assets/165a3f18-659a-4e2e-a154-a3456b667bae)


These visualizations show how F0 (fundamental frequency/pitch) information affects the model's rotary position embeddings (RoPE)


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


These visualizations help confirm that:
- F0 information is being properly integrated into the model
- The adaptation creates meaningful variations in the position encodings
- The signal is strong enough to potentially help the model understand pitch-sensitive aspects of speech
  
----
#### Domain-Specific ASR model. 

#### freqs = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000

#### Static frequency's are perfectly fine for text models but not for NLP.

(https://huggingface.co/Sin2pi/Echo17/tensorboard?params=scalars)

----

The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

<img width="470" alt="cc" src="https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8"  />
<img width="470" alt="cc2" src="https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1"  />


Echos rotary implementation maps the perceptual properties of audio to the mathematical properties of the rotary embeddings, creating a more adaptive and context-aware representation system. Pitch is optionally extracted from audio in the data processing pipeline and can be used for an additional feature along with spectrograms and or used to inform the rotary and or pitch bias.

Pitch bias

The pitch bias implementation creates an attention bias matrix:
This makes tokens with similar pitch attend to each other more, which helps:

- Track speaker consistency
- Maintain coherent pitch patterns
- Group harmonically related segments

The theoretical foundation:
- Both position and pitch can be represented as frequencies
- Speech has inherent rhythmic and tonal patterns that correlate with semantic content
- Varying the rotation frequency based on pitch creates a more speech-aware positional encoding

--- 


