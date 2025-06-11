## Echo - NLP/ASR model with acoustic Rotary Position Encoding (vRoPE).  And some other stuff...

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

<img width="570" alt="score" src="https://github.com/user-attachments/assets/679d5032-6e84-4fe6-892c-6b01c6cb14ce" />

📊 COMPONENT STATISTICS:
  GATE: avg=0.638041, min=0.010094, max=2.071990, samples=135
  MLP: avg=0.028625, min=0.003352, max=0.074448, samples=135
  Q: avg=0.029973, min=0.001905, max=0.141696, samples=150
  K: avg=0.030055, min=0.001910, max=0.144063, samples=150
  V: avg=0.111713, min=0.050426, max=0.240650, samples=150
  O: avg=0.108549, min=0.049052, max=0.244606, samples=150
  LN: avg=0.092093, min=0.005017, max=0.349827, samples=285
  ENCODER: avg=0.004097, min=0.001447, max=0.011093, samples=45

🚨 GATE vs MLP ACTIVATION PATTERNS:
🟢 encoder.blocks.spectrogram.1.: gate/mlp activation ratio=1.4918, sparsity difference=-0.0040
🟡 encoder.blocks.spectrogram.2.: gate/mlp activation ratio=2.5671, sparsity difference=-0.0096
🟢 encoder.blocks.spectrogram.3.: gate/mlp activation ratio=1.9277, sparsity difference=-0.0069
🟡 encoder.blocks.spectrogram.4.: gate/mlp activation ratio=2.6485, sparsity difference=-0.0118
🟡 decoder._blocks.0.: gate/mlp activation ratio=2.0988, sparsity difference=-0.0071
🟡 decoder._blocks.1.: gate/mlp activation ratio=2.1584, sparsity difference=-0.0102
🟡 decoder._blocks.2.: gate/mlp activation ratio=2.1087, sparsity difference=-0.0096
🟡 decoder._blocks.3.: gate/mlp activation ratio=2.2582, sparsity difference=-0.0045
🟡 decoder.blocks.spectrogram.0.: gate/mlp activation ratio=2.0964, sparsity difference=-0.0124
🟢 decoder.blocks.spectrogram.1.: gate/mlp activation ratio=1.9247, sparsity difference=-0.0021
🟢 decoder.blocks.spectrogram.2.: gate/mlp activation ratio=1.8573, sparsity difference=-0.0079
🟢 decoder.blocks.spectrogram.3.: gate/mlp activation ratio=1.8911, sparsity difference=-0.0062


## The F0-Conditioned Rotation Mechanism

The high gate usage validates the fundamental frequency conditioning approach:

- Pitch-adaptive rotary embeddings are providing meaningful signal that the gates are actively utilizing
- The decoder is learning to selectively attend to pitch-relevant patterns
- The gates are functioning as a kind of "pitch-aware filter" that determines which information should flow through the network


![sp](https://github.com/user-attachments/assets/a29f8c97-71c7-4bfc-9c11-76005614822c)
