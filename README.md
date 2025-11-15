
        The model doesn't need many layers or dims to converge. Tested thoroughly for convergence. 
        Not sure about generalization since I only protoype. 
        
        This is version 100k or something. Since whisper was released I realized there was a problem 
        either with openai code or ASR models in general. Can't be fined tuned without catastrophic 
        forgetting that was reversable in any meaningful way. Finetuning events were overwriting 
        whatever the model had learned before. 
        So either overwrite less or get the model to forget less or both. 
        
        Feature injection at the embedding level gives the model recurrent-like qualities without the baggage. 
        Uses log mels, pitch, waveform (and others, pitch-tokens, hilberts and harmonics), 
        This specific version uses log-mels, pitch, waveform, and pitch-tokens but you can 
        toggle them on and off for experimenting. Some other tricks.
        
        Not everything in the script is fully integrated yet. 
        I'm still trying to think through best how to inject characteristics that are unique to each
        sound into the rotary without breaking things too badly. 
        Might go old gangster next and use some kind of genralized frequency phenom map as radius since its 
        consitent enough while still being able to apply it from f0 of the samples. 
        Mix that with some set theta decay. The idea is to change the pairwise 
        rotations with just enough real data to make it meaningful but not enough to break its functioning. 
        
        How to use:
        install any imports in the imports section you dont have.
        
        This script uses a hugging face hosted dataset so you might need your token or just use a different dataset.
        
        python model.py

        I suggest using Maxfactor (not Maxfactor2) in optimizer.py but a standard adam or adafactor works well too.
