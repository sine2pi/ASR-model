## Adaptive Audio Features

Pitch/frequency/waveform/spectrogram are options.

The F0 can be used together to analyze the prosody of speech, including intonation and loudness. The F0 contour follows the lowest frequency with the most energy, which is indicated by bright colors towards the bottom of the image. 
F0 contour represents pitch variation over time, while energy contour represents sound intensity across frequencies over time. They both play a crucial role in understanding speech prosody and can be used together to analyze emotional expressions and grammatical structures within speech. 

I've combined these (and periodicity) as an optional pitch feature while f0 can be optionally mapped to the rotary embedding theta value / per step during training.

This model optionally learns each audio feature in seperate layers in sequence each feature building on the other. Like human reinforcment learning, the model can change the order of feature learned. 

