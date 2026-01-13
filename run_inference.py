import torch
import argparse
from modelnew import Model, Dimensions
from essentials import prepare_datasets, DataCollator, setup_tokenizer

def run_inference(args):
    """
    Runs inference on a single data sample using a trained ASR model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = setup_tokenizer(args.tokenizer_path)
    print("Tokenizer loaded.")

    param = Dimensions(
        tokens=40000, 
        mels=128, 
        dims=512, 
        head=4, 
        layer=4, 
        act="gelu", 
        n_type="layernorm"
    )

    model = Model(param).to(device)
    
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print(f"Model checkpoint loaded successfully from {args.checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    model.eval()

    extract_args = {
        "spectrogram": False,
        "pitch": True,
        "waveform": False,
        "pitch_tokens": False,
        "hop_length": 160,
        "sample_rate": 16000,
        "mels": 128
    }

    dataset = prepare_datasets(args.metadata_file, args.data_dir, tokenizer, extract_args=extract_args)
    collator = DataCollator(tokenizer=tokenizer)

    if args.sample_index >= len(dataset):
        print(f"Error: Sample index {args.sample_index} is out of bounds for dataset of size {len(dataset)}.")
        return
        
    sample = dataset[args.sample_index]
    batch = collator([sample])

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
            
    print("\nRunning inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            spectrogram=batch.get("spectrogram"),
            pitch=batch.get("pitch"),
            waveform=batch.get("waveform"),
            pitch_tokens=batch.get("pitch_tokens"),
            max_new_tokens=args.max_new_tokens
        )

    ground_truth_text = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    
    print(f"\nRaw Generated Token IDs: {generated_ids[0].cpu().numpy().tolist()}")

    predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("-" * 50)
    print(f"Sample Index: {args.sample_index}")
    print(f"Ground Truth: {ground_truth_text[0]}")
    print(f"Prediction:   {predicted_text[0]}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR inference on a single sample.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--metadata_file", type=str, default="./LJSpeech1000/metadata.csv",
                        help="Path to the dataset metadata file.")
    parser.add_argument("--data_dir", type=str, default="./LJSpeech1000",
                        help="Directory containing the audio files.")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json",
                        help="Path to the tokenizer.json file.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="The index of the sample in the dataset to test.")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Maximum number of new tokens to generate.")

    args = parser.parse_args()
    run_inference(args)

