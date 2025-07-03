import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from file import process_file
import argparse
import os
from prompts import transcribe_prompt

USE_VAD = False
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def transcribe_file(filename, model, processor, use_vad=USE_VAD):
    start_time = time.time()
    all_outputs = process_file(filename, model, processor, transcribe_prompt, use_vad)
    elapsed_time = time.time() - start_time
    return all_outputs, elapsed_time

def main():
    parser = argparse.ArgumentParser(
        description="Process an input file and output directory."
    )

    parser.add_argument(
        "input_file", type=str, help="Path to the input file."
    )

    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to the output directory."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["google/gemma-3n-E4B-it", "google/gemma-3n-E2B-it"],
        default="google/gemma-3n-E4B-it",
        help="Model to use: 'google/gemma-3n-E4B-it' or 'google/gemma-3n-E2B-it'",
    )

    args = parser.parse_args()

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", device_map="cuda"
    )

    results, elapsed_time = transcribe_file(args.input_file, model, processor)
    print(f"Total time used: {elapsed_time:.2f} seconds")

    audio_basename = os.path.basename(args.input_file)
    audio_basename = os.path.splitext(audio_basename)[0]
    output_path = os.path.join(args.output_dir, audio_basename + ".txt")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w") as outfile:
        for result in results:
            outfile.write(result + "\n")

    print(f"File saved to {output_path}")

if __name__ == "__main__":
    main()
