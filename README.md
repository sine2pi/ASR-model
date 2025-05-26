## Adaptive Audio Features

Pitch/frequency/waveform/spectrogram are options.

The model combines these as optional features while f0 can be optionally mapped to the rotary embedding theta value / per step during training.

This model can earn each audio feature in seperate layers in sequence each feature building on the other. The order in which these features are learned is dynamic.
