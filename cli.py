import time
import argparse
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from file import process_file
from prompts import transcribe_prompt


def load_model_and_processor(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda"
    )
    return model, processor


def transcribe_file(filename, model, processor, prompt, temperature, use_vad=False):
    start_time = time.time()
    all_outputs = process_file(filename, model, processor, prompt, use_vad, temperature)
    elapsed_time = time.time() - start_time
    return all_outputs, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")

    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["google/gemma-3n-E4B-it", "google/gemma-3n-E2B-it"],
        default="google/gemma-3n-E4B-it",
        help="Model to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=transcribe_prompt,
        help="Prompt to guide transcription.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for decoding."
    )
    parser.add_argument(
        "--use_vad", action="store_true", help="Enable voice activity detection (VAD)."
    )

    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model)
    results, elapsed_time = transcribe_file(
        args.input_file, model, processor, args.prompt, args.temperature, args.use_vad
    )

    print(f"Total time used: {elapsed_time:.2f} seconds")

    os.makedirs(args.output_dir, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(args.output_dir, audio_basename + ".txt")

    with open(output_path, "w") as f:
        for result in results:
            f.write(result + "\n")

    print(f"File saved to {output_path}")


if __name__ == "__main__":
    main()
