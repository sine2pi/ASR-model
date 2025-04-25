ASR encoder-decoder model with optional blending of spectrogram and waveform input. Full script with tranining loop compatable with hugging face. For testing.

This model's learnable blend (with a sigmoid-mixed parameter) between waveform and spectrogram encodings is a novel and practical way to let the model decide the optimal mix. This form of adaptive fusion is less common in open-source ASR codebases.

Blending waveform and spectrogram features has been explored in some research, but is not standard in ASR pipelines.
This learnable blend is a modern, under-explored approach addressing the waveform spectrogram debate. Initial findings of the pilot run suggest that the blending of the two significantly decreases WER compared to standalone waveform and spectrogram without significantly increasing overhead. 

3 tests: spectrogram, waveform, spectrogram+waveform.
### IN PROGRESS

