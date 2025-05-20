## Zero-Value Processing in Speech Attention

This model learns each audio feature in seperate layers in sequence each feature building on the other. Like human reinforcment learning. You can change the order of feature learned. Waveform spectrogram and pitch.

### The Significance of Zeros in Audio Processing

In log-mel spectrograms, zero or near-zero values represent critical information:
- Silent regions between speech
- Low-amplitude acoustic events
- Sub-threshold background environments

The model also extracts f0 contour and energy contours.
We use them as features and we then inject the frequency into the rotary at the same time as the sample hits the multihead and encoder. Frerquency is a learnable parameter.

### Multiplicative Soft Masking: Technical Implementation

1. **Semantic Preservation of Silence**: Unlike conventional `-inf` masking that eliminates attention, this approach maintains minimal attention flow (0.000001) for silence tokens, preserving their semantic value.

2. **Prosodic Pattern Recognition**: By allowing minimal attention to silent regions, the model can learn timing, rhythm, and prosodic features critical for speech understanding.

3. **Cross-Modal Representation Unification**: Using identical attention mechanisms for both audio silence and text padding creates a unified approach across modalities, simplifying the architecture.

4. **Training Stability**: This multiplicative masking approach maintains gradient flow through all positions, potentially improving convergence stability.

5. **Natural Boundary Learning**: By processing silence regions with reduced but non-zero attention, the model learns natural speech boundaries without requiring explicit BOS/EOS tokens.

- Force the model to not rely on start tokens
- Create a cleaner boundary between "meaningful" and "non-meaningful" tokens
- Simplify the attention mechanism's behavior

This explicitly creates a system where:
- Zero-valued tokens get minimal attention (multiplied by 0.000001)
- All non-zero tokens get normal attention weights

1. **All padding uses 0** - Which gets minimal attention (good)
2. **All start tokens use 0** - Also gets minimal attention (intentional)
3. **All meaningful content uses non-zero tokens** - Gets normal attention (good)

## Potential Benefits

This approach could:
- Force the model to not rely on start tokens
- Create a cleaner boundary between "meaningful" and "non-meaningful" tokens
- Simplify the attention mechanism's behavior

The critical question is: Does a model need the start token to initialize proper generation? In most transformer models, the first token (often a start token) provides essential context for generating the first meaningful token.

This design intentionally minimizes the start token's influence.. if the model performs well, then this could be a novel and interesting approach to sequence generation.

## Adaptive Audio Features

Pitch/frequency/waveform/spectrogram are options.

The F0 contour and the energy contour can be used together to analyze the prosody of speech, including intonation and loudness. The F0 contour follows the lowest frequency with the most energy, which is indicated by bright colors towards the bottom of the image. 
F0 contour represents pitch variation over time, while energy contour represents sound intensity across frequencies over time. They both play a crucial role in understanding speech prosody and can be used together to analyze emotional expressions and grammatical structures within speech. 

I've combined these (and periodicity) as an optional pitch feature while f0 can be optionally mapped to the rotary embedding theta value / per step during training.

This model learns each audio feature in seperate layers in sequence each feature building on the other. Like human reinforcment learning. You can change the order of feature learned. 


