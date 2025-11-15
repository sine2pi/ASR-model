        The model doesn't need many layers or dims to converge. Tested thoroughly for convergence. Not sure about generalization since I only protoype. This         is version 100k or something. Since whisper was released I realized there was a problem either with openai code or ASR models in general. Can't              be fined tuned without catastrophic forgetting that was reversable in any meaningful way. Finetuning events were overwriting whatever the model              had learned before. So either overwrite less or get the model to forget less or both. 
        
        f0 injection at the embedding level gives the model recurrent-like qualities without the baggage. Uses log mels, pitch, waveform (and others, 
        pitch-tokens, hilberts and harmonics), This specific version uses log-mels, pitch, waveform, and pitch-tokens but you can togle them on and off for          experimenting. Some other tricks.
        
        Not everything in the script is fully integrated yet. 
        
        How to use:
        install any imports in the imports section you dont have.
        
        This script uses a hugging face hosted dataset so you might need your token or just use a different dataset.
        
        python model.py

        I suggest using Maxfactor (not Maxfactor2) in optimizer.py but a standard adam or adafactor works well too.


