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

### Diagnostic test run with google/fleurs - Spectrogram + f0_rotary:

![123](https://github.com/user-attachments/assets/8eb4146b-2dfe-4e93-9f14-789ac5f5d3af)

![flow](https://github.com/user-attachments/assets/28298306-816d-40b5-8390-63c762e0b69f)

This was a pilot run for diagnostics and it's too early to make any statement about the model.

