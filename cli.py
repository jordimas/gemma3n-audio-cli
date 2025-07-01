import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from file import process_file
import argparse
import os
from prompts import transcribe_prompt

GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"
USE_VAD = False


os.environ["TRANSFORMERS_OFFLINE"] = "1"


processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="auto")
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID, torch_dtype="auto", device_map="auto"
)


def transcribe_file(filename, use_vad=USE_VAD):
    start_time = time.time()
    all_outputs = process_file(filename, model, processor, transcribe_prompt, use_vad)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print total time used
    return all_outputs, elapsed_time


def main():
    parser = argparse.ArgumentParser(
        description="Process an input file and output directory."
    )

    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to the output directory."
    )

    parser.add_argument("input_file", type=str, help="Path to the input file.")

    args = parser.parse_args()

    results, elapsed_time = transcribe_file(args.input_file)
    print(f"Total time used: {elapsed_time:.2f} seconds")

    audio_basename = os.path.basename(args.input_file)
    audio_basename = os.path.splitext(audio_basename)[0]
    output_path = os.path.join(args.output_dir, audio_basename + "." + "txt")

    with open(output_path, "w") as outfile:
        for result in results:
            outfile.write(result + "\n")

    print(f"File copied to {output_path}")


if __name__ == "__main__":
    main()
