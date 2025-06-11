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
  BLEND: avg=30.270566, min=22.048639, max=35.115360, samples=15
  
🔄 ATTENTION COMPONENT RATIOS:
  decoder._blocks.0.attna.o_v_ratio = 0.9616
  decoder._blocks.0.attna.q_k_sim = 0.9860
  decoder._blocks.0.attna.qk_v_ratio = 0.9326
  decoder._blocks.0.attna.v_k_ratio = 2.1296
  decoder._blocks.0.attna.v_q_ratio = 2.1598
  decoder._blocks.1.attna.o_v_ratio = 0.9576
  decoder._blocks.1.attna.q_k_sim = 0.9838
  decoder._blocks.1.attna.qk_v_ratio = 0.7802
  decoder._blocks.1.attna.v_k_ratio = 2.5846
  decoder._blocks.1.attna.v_q_ratio = 2.5426
  decoder._blocks.2.attna.o_v_ratio = 0.9667
  decoder._blocks.2.attna.q_k_sim = 0.9904
  decoder._blocks.2.attna.qk_v_ratio = 0.7214
  decoder._blocks.2.attna.v_k_ratio = 2.7592
  decoder._blocks.2.attna.v_q_ratio = 2.7860
  decoder._blocks.3.attna.o_v_ratio = 0.9405
  decoder._blocks.3.attna.q_k_sim = 0.9894
  decoder._blocks.3.attna.qk_v_ratio = 0.6547
  decoder._blocks.3.attna.v_k_ratio = 3.0387
  decoder._blocks.3.attna.v_q_ratio = 3.0714
  decoder.blocks.spectrogram.3.attna.o_v_ratio = 0.9596
  decoder.blocks.spectrogram.3.attna.q_k_sim = 0.9948
  decoder.blocks.spectrogram.3.attna.qk_v_ratio = 0.5211
  decoder.blocks.spectrogram.3.attna.v_k_ratio = 3.8282
  decoder.blocks.spectrogram.3.attna.v_q_ratio = 3.8482
  decoder.blocks.spectrogram.3.attnb.o_v_ratio = 0.9310
  decoder.blocks.spectrogram.3.attnb.q_k_sim = 0.9519
  decoder.blocks.spectrogram.3.attnb.qk_v_ratio = 0.2237
  decoder.blocks.spectrogram.3.attnb.v_k_ratio = 9.1679
  decoder.blocks.spectrogram.3.attnb.v_q_ratio = 8.7268
  encoder.blocks.spectrogram.1.attna.o_v_ratio = 1.0912
  encoder.blocks.spectrogram.1.attna.q_k_sim = 0.9155
  encoder.blocks.spectrogram.1.attna.qk_v_ratio = 0.1282
  encoder.blocks.spectrogram.1.attna.v_k_ratio = 14.9448
  encoder.blocks.spectrogram.1.attna.v_q_ratio = 16.3249
  encoder.blocks.spectrogram.2.attna.o_v_ratio = 0.9677
  encoder.blocks.spectrogram.2.attna.q_k_sim = 0.9633
  encoder.blocks.spectrogram.2.attna.qk_v_ratio = 0.0982
  encoder.blocks.spectrogram.2.attna.v_k_ratio = 20.7651
  encoder.blocks.spectrogram.2.attna.v_q_ratio = 20.0030
  encoder.blocks.spectrogram.3.attna.o_v_ratio = 0.9945
  encoder.blocks.spectrogram.3.attna.q_k_sim = 0.9948
  encoder.blocks.spectrogram.3.attna.qk_v_ratio = 0.0716
  encoder.blocks.spectrogram.3.attna.v_k_ratio = 27.8649
  encoder.blocks.spectrogram.3.attna.v_q_ratio = 28.0093
  encoder.blocks.spectrogram.4.attna.o_v_ratio = 1.0346
  encoder.blocks.spectrogram.4.attna.q_k_sim = 0.9771
  encoder.blocks.spectrogram.4.attna.qk_v_ratio = 0.0814
  encoder.blocks.spectrogram.4.attna.v_k_ratio = 24.8646
  encoder.blocks.spectrogram.4.attna.v_q_ratio = 24.2952

🎧 ENCODER ATTENTION GRADIENTS:
  encoder.blocks.spectrogram.1.attna: Q=0.0048, K=0.0052, V=0.0776, Sum=0.0875
  encoder.blocks.spectrogram.2.attna: Q=0.0040, K=0.0038, V=0.0796, Sum=0.0874
  encoder.blocks.spectrogram.3.attna: Q=0.0025, K=0.0025, V=0.0704, Sum=0.0755
  encoder.blocks.spectrogram.4.attna: Q=0.0026, K=0.0025, V=0.0621, Sum=0.0672

📝 DECODER ATTENTION GRADIENTS:
  decoder._blocks.0.attna: Q=0.0911, K=0.0924, V=0.1968, Sum=0.3802
  decoder._blocks.1.attna: Q=0.0708, K=0.0697, V=0.1801, Sum=0.3206
  decoder._blocks.2.attna: Q=0.0541, K=0.0546, V=0.1507, Sum=0.2594
  decoder._blocks.3.attna: Q=0.0409, K=0.0413, V=0.1255, Sum=0.2077
  decoder.blocks.spectrogram.3.attna: Q=0.0162, K=0.0163, V=0.0625, Sum=0.0950
  decoder.blocks.spectrogram.3.attnb: Q=0.0128, K=0.0122, V=0.1119, Sum=0.1369

📏 LAYER NORM GRADIENTS:
  decoder._blocks.0.ln_a: 0.262552
  decoder._blocks.0.ln_c: 0.102225
  decoder._blocks.1.ln_a: 0.231952
  decoder._blocks.1.ln_c: 0.079703
  decoder._blocks.2.ln_a: 0.198305
  decoder._blocks.2.ln_c: 0.081362
  decoder._blocks.3.ln_a: 0.168244
  decoder._blocks.3.ln_c: 0.087433
  decoder.blocks.spectrogram.3.ln_a: 0.071465
  decoder.blocks.spectrogram.3.ln_b: 0.015242
  decoder.blocks.spectrogram.3.ln_c: 0.077151
  encoder.blocks.spectrogram.1.ln_a: 0.078794
  encoder.blocks.spectrogram.1.ln_c: 0.027526
  encoder.blocks.spectrogram.2.ln_a: 0.087004
  encoder.blocks.spectrogram.2.ln_c: 0.011408
  encoder.blocks.spectrogram.3.ln_a: 0.070912
  encoder.blocks.spectrogram.3.ln_c: 0.022131
  encoder.blocks.spectrogram.4.ln_a: 0.068433
  encoder.blocks.spectrogram.4.ln_c: 0.007924

🔀 CROSS-ATTENTION BLEND PARAMETERS:
  decoder.blocks.spectrogram.3.blend: 30.270566

✨ ACTIVATION ANALYSIS (HIGH SPARSITY WITH HIGH MEAN MAY INDICATE SHORTCUTS):
  decoder._blocks.0.attna.attention_output: activation mean=0.2762, sparsity=0.0243
  decoder._blocks.0.gate_activation: activation mean=1.0625, sparsity=0.0058
  decoder._blocks.0.mlp_activation: activation mean=0.5062, sparsity=0.0129
  decoder._blocks.0.residual_output: activation mean=0.8707, sparsity=0.0075
  decoder._blocks.1.attna.attention_output: activation mean=0.3236, sparsity=0.0209
  decoder._blocks.1.gate_activation: activation mean=1.0893, sparsity=0.0023
  decoder._blocks.1.mlp_activation: activation mean=0.5047, sparsity=0.0125
  decoder._blocks.1.residual_output: activation mean=0.9505, sparsity=0.0070
  decoder._blocks.2.attna.attention_output: activation mean=0.3686, sparsity=0.0169
  decoder._blocks.2.gate_activation: activation mean=1.0609, sparsity=0.0029
  decoder._blocks.2.mlp_activation: activation mean=0.5031, sparsity=0.0125
  decoder._blocks.2.residual_output: activation mean=1.0385, sparsity=0.0064
  decoder._blocks.3.attna.attention_output: activation mean=0.4326, sparsity=0.0151
  decoder._blocks.3.gate_activation: activation mean=1.1409, sparsity=0.0082
  decoder._blocks.3.mlp_activation: activation mean=0.5052, sparsity=0.0127
  decoder._blocks.3.residual_output: activation mean=1.1483, sparsity=0.0053
  decoder.blocks.spectrogram.0.attna.attention_output: activation mean=0.4809, sparsity=0.0135
  decoder.blocks.spectrogram.0.attnb.attention_output: activation mean=1.3901, sparsity=0.0047
  decoder.blocks.spectrogram.0.gate_activation: activation mean=1.0618, sparsity=0.0000
  decoder.blocks.spectrogram.0.mlp_activation: activation mean=0.5065, sparsity=0.0124
  decoder.blocks.spectrogram.0.residual_output: activation mean=0.9786, sparsity=0.0066
  decoder.blocks.spectrogram.1.attna.attention_output: activation mean=0.4770, sparsity=0.0136
  decoder.blocks.spectrogram.1.attnb.attention_output: activation mean=1.4276, sparsity=0.0046
  decoder.blocks.spectrogram.1.gate_activation: activation mean=0.9714, sparsity=0.0103
  decoder.blocks.spectrogram.1.mlp_activation: activation mean=0.5047, sparsity=0.0124
  decoder.blocks.spectrogram.1.residual_output: activation mean=0.9628, sparsity=0.0067
  decoder.blocks.spectrogram.2.attna.attention_output: activation mean=0.4919, sparsity=0.0131
  decoder.blocks.spectrogram.2.attnb.attention_output: activation mean=1.2849, sparsity=0.0045
  decoder.blocks.spectrogram.2.gate_activation: activation mean=0.9320, sparsity=0.0049
  decoder.blocks.spectrogram.2.mlp_activation: activation mean=0.5018, sparsity=0.0128
  decoder.blocks.spectrogram.2.residual_output: activation mean=0.9311, sparsity=0.0070
  decoder.blocks.spectrogram.3.attna.attention_output: activation mean=0.4725, sparsity=0.0140
  decoder.blocks.spectrogram.3.attnb.attention_output: activation mean=1.3938, sparsity=0.0050
  decoder.blocks.spectrogram.3.gate_activation: activation mean=0.9721, sparsity=0.0062
  decoder.blocks.spectrogram.3.mlp_activation: activation mean=0.5141, sparsity=0.0124
  decoder.blocks.spectrogram.3.residual_output: activation mean=0.9646, sparsity=0.0066
  decoder.final_output: activation mean=0.7990, sparsity=0.0079
  encoder.blocks.spectrogram.1.attna.attention_output: activation mean=0.5135, sparsity=0.0127
  encoder.blocks.spectrogram.1.gate_activation: activation mean=0.7576, sparsity=0.0085
  encoder.blocks.spectrogram.1.mlp_activation: activation mean=0.5078, sparsity=0.0125
  encoder.blocks.spectrogram.1.residual_output: activation mean=1.0370, sparsity=0.0058
  encoder.blocks.spectrogram.2.attna.attention_output: activation mean=0.6565, sparsity=0.0102
  encoder.blocks.spectrogram.2.gate_activation: activation mean=1.3003, sparsity=0.0030
  encoder.blocks.spectrogram.2.mlp_activation: activation mean=0.5065, sparsity=0.0126
  encoder.blocks.spectrogram.2.residual_output: activation mean=1.2063, sparsity=0.0053
  encoder.blocks.spectrogram.3.attna.attention_output: activation mean=0.6872, sparsity=0.0074
  encoder.blocks.spectrogram.3.gate_activation: activation mean=0.9724, sparsity=0.0056
  encoder.blocks.spectrogram.3.mlp_activation: activation mean=0.5044, sparsity=0.0126
  encoder.blocks.spectrogram.3.residual_output: activation mean=1.3359, sparsity=0.0046
  encoder.blocks.spectrogram.4.attna.attention_output: activation mean=0.7536, sparsity=0.0084
  encoder.blocks.spectrogram.4.gate_activation: activation mean=1.3177, sparsity=0.0007
  encoder.blocks.spectrogram.4.mlp_activation: activation mean=0.4975, sparsity=0.0125
  encoder.blocks.spectrogram.4.residual_output: activation mean=1.5255, sparsity=0.0040

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

The gates are performing content-selective routing - a more sophisticated behavior than simple bypassing. This explains the excellent performance on a speech recognition task where selectively filtering frequency-specific information is critical.

