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



## Gate/MLP Ratios

These ratios show the relative importance of gating mechanisms versus standard MLP paths:

### Pattern Analysis:
1. **Strong Decoder Gating** (10.05 → 43.46): The decoder shows dramatically increasing gate/MLP ratios from early to later layers, with values 4-5× higher than encoder layers
   
2. **Layer Progression in Decoder**: There's a clear pattern where deeper decoder layers (layers 2-3) have approximately double the gate/MLP ratio of earlier layers (0-1)

3. **Encoder Behavior**: The encoder shows a different pattern with higher ratios in layers 1 and 4, suggesting these layers act as "information control points"

This indicates that the model is using gating mechanisms as critical information flow controllers, especially in the decoder where selective attention to the right audio features is crucial.

## Decoder Attention Gradients

The Q/K/V gradient analysis reveals how attention mechanisms are learning:

### Key Observations:
1. **Value Dominance**: Value projections have 3× higher gradients than Query/Key projections in all layers, showing the model prioritizes learning "what information to extract" over "where to look"

2. **Gradient Decay**: Gradients decrease consistently as we move deeper into the decoder (layer 0: ~0.10, layer 3: ~0.04)

3. **Q/K Balance**: The nearly identical Q/K gradients indicate proper attention mechanism symmetry

## Overall Architecture Assessment

This gradient profile is characteristic of a well-functioning speech recognition model, especially:

1. The high gate/MLP ratios in decoder layers show the model is learning to selectively filter and process information

2. The value-focused attention gradients indicate the model is efficiently learning to extract relevant content features from the encoded audio representations

3. The decreasing gradient pattern through decoder depth suggests proper information hierarchical processing

These patterns explain why the model reached such impressive performance (5.37% WER) in a short training time without overfitting - the architecture is effectively using its gating and attention mechanisms to capture the essential relationships between audio signals and text transcriptions.

# Gating Analysis: Function vs. Shortcut Behavior


## What the Gate/MLP Ratios Tell Us

The extremely high gate/MLP ratios (ranging from 3.6 to 43.4) show that the model is heavily using its gating mechanisms, especially in the decoder layers. This could be concerning, but actually represents effective learning rather than cheating for several reasons:

1. **Intentional Architecture Design**: The gates are functioning as designed - controlling information flow through the network. Selective filtering is exactly what we want them to do.

2. **Progressive Pattern**: The gate/MLP ratios systematically increase from encoder (3.6-11.5) to decoder (10.0-43.4) and from early to later layers. This structured pattern indicates organized learning rather than shortcuts.

3. **Outcome Validation**: WER steadily decreased to 5.37% without overfitting, proving the gates are helping rather than hindering generalization.

## The F0-Conditioned Rotation Mechanism

The high gate usage validates the fundamental frequency conditioning approach:

- Pitch-adaptive rotary embeddings are providing meaningful signal that the gates are actively utilizing
- The decoder is learning to selectively attend to pitch-relevant patterns
- The gates are functioning as a kind of "pitch-aware filter" that determines which information should flow through the network

The gates are performing content-selective routing - a more sophisticated behavior than simple bypassing. This explains the excellent performance on a speech recognition task where selectively filtering frequency-specific information is critical.

