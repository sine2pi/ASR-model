### NLP/ASR model with acoustic variable radii encoding (vRoPE). 

To highlight the relationship between pitch and rotary embeddings echo implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)
  -- Tests indicate that direct use of f0 without mapping resulted in better WER, 10k arbitrary?
3. The second adds direct similarity bias to attention
4. Variable radii added in place of unit circle radius(1.0) associated with torch.polar. The frequencies (f0) are time aligned with tokens creating acoustically-weighted positional encodings where the "loudness" of each position in the embedding space reflects the acoustic prominence in the original speech.

By modulating the RoPE frequencies based on pitch (F0), we are essentially telling the model to pay attention to the acoustic features relate to sequence position in a way that's proportional to the voice characteristics.  This approach creates a more speech-aware positional representation that helps the model better understand the relationship between acoustic features and text.

1000 steps no f0:

<img width="470" alt="123" src="https://github.com/user-attachments/assets/1b3ca1e8-0b7d-47dd-802b-5eda9537ae13" />

1000 steps with f0 / theta substitutions:

<img width="470" alt="321" src="https://github.com/user-attachments/assets/24a68910-b316-4cfc-8927-5c6fd846b919" />



The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

![2](https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8)
![1](https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1)

The rotary implementation maps the perceptual properties of audio to the mathematical properties of the rotary embeddings, creating a more adaptive and context-aware representation system. Pitch is optionally extracted from audio in the data processing pipeline and can be used for an additional feature along with spectrograms and or used to inform the rotary and or pitch bias.

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

This rotary also uses variable radii. Pitch maps to each via a variable length radius adding a dimension of power or magnitutde to standard RoPE.

--- 

### Diagnostic test run with google/fleurs - Spectrogram + f0_rotary:

<img width="689" alt="graph" src="https://github.com/user-attachments/assets/c161a89d-539c-4983-8d24-12ec41ebc859" />
<img width="277" alt="321" src="https://github.com/user-attachments/assets/4cc71b43-3e48-4241-b381-5bda17ed9d0d" />
<img width="727" alt="eff" src="https://github.com/user-attachments/assets/ffb9dd2f-e536-4d4d-9590-cacc1e78ebcf" />

https://huggingface.co/Sin2pi/Echo17/tensorboard


## The F0-Conditioned Rotation Mechanism

The high gate usage validates the fundamental frequency conditioning approach:

- Pitch-adaptive rotary embeddings are providing meaningful signal that the gates are actively utilizing
- The decoder is learning to selectively attend to pitch-relevant patterns
- The gates are functioning as a kind of "pitch-aware filter" that determines which information should flow through the network
- Mapping f0 to standard 10k theta/base results worse than using f0 directly.





