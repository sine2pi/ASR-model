## Echo
### Zero-Value Processing ASR model with Voice-modulated Rotary Position Encoding. (vRoPE)

To highlight the relationship between pitch and rotary embeddings echo implements two complementary pitch-based enhancements:

1. The first uses pitch to modify theta (rotary frequency)
2. The second adds direct similarity bias to attention

By modulating the RoPE frequencies based on pitch (F0), we are essentially telling the model to pay attention to the acoustic features relate to sequence position in a way that's proportional to the voice characteristics.  This approach creates a more speech-aware positional representation that helps the model better understand the relationship between acoustic features and text.

The patterns below show how positions "see" each other in relation to theta and f0. 

Bright diagonal line: Each position matches itself perfectly.
Wider bright bands: Positions can "see" farther (good for long dependencies) but can be noisy.
Narrow bands: More focus on nearby positions (good for local patterns)

![2](https://github.com/user-attachments/assets/28d00fc5-2676-41ed-a971-e4d857af43f8)
![1](https://github.com/user-attachments/assets/9089e806-966b-41aa-8793-bee03a6e6be1)

Static 10k theta is perfectly fine for a text model but probably not for a NLP ai.

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

Leveraging Silence for Natural Generation Stopping
By scaling attention scores related to pad/silence tokens down to near zero, we are creating a consistent pattern that the model can learn

This creates a direct correspondence between:
- Log mel spectrograms near zero in silent regions
- Attention scores that are scaled down for pad tokens

The benefits of this approach:

1. Consistent signals: The model gets similar signals for silence in both input data and attention patterns
2. Content-based stopping: Model can learn to end generation based on acoustic properties rather than just position
3. Learnable behavior: Using a learnable parameter (`self.fzero`) lets the model find the optimal scaling factor

1. Zero in loss calculation
   The model explicitly ignores index 0 in the loss calculation using `ignore_index=0` parameter.
2. Zero for special tokens
3. Attention scaling for zero tokens
   This specifically scales attention scores for pad tokens (0) down to near-zero values.

In standard usage, RoPE encodes relative positional information by applying frequency-based rotations to token embeddings. What this does is creates a meaningful bridge between two domains:

1. Token positions (sequential information)
2. Pitch information (acoustic properties)


