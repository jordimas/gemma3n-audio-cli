import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from file import process_file
import argparse
import os

GEMMA_MODEL_ID = "google/gemma-3n-E2B-it"

os.environ["TRANSFORMERS_OFFLINE"] = "1"


processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="")
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID, torch_dtype="auto", device_map="cpu"
)

identify_lang_prompt = {
    "type": "text",
    "text": "Identify the languages of the audio and generate a JSON with the following fields\n"
    "- language: a iso-639 language identified\n"
    "- confidence level from 0 to 1 of the language identified\n",
}


transcribe_prompt_works = {
    "type": "text",
    "text": "You are given an audio recording in Catalan language with multiple speakers. Your task is to separate the speech segments by speaker and assign a unique speaker label to each segment. Output the start time, end time, and speaker ID for each segment."
    "Format:\n"
    "[start_time - end_time] Speaker_ID: Transcription (optional)\n"
    "\n"
    "Example:\n"
    "[00:00:00 - 00:00:05] Speaker_1: Hello, how are you?\n"
    "[00:00:06 - 00:00:10] Speaker_2: I'm good, thanks!\n"
    "\n"
    "Please provide the diarization results.\n",
}

transcribe_prompt = {
    "type": "text",
    "text": "You are given an audio recording in Catalan language with multiple speakers. Your task is to separate the speech segments by speaker and assign a unique speaker label to each segment. Output the start time, end time, and speaker ID for each segment and gender.",
    # "Format:\n"
    # "[start_time - end_time] Speaker_ID: Transcription\n"
}


# Does not seem to honor JSON
transcribe_prompt = {"type": "text", "text": "Transcribe the audio file in Catalan.\n"}


def transcribe_file(filename):
    start_time = time.time()

    # filename = "dosparlants.mp3"
    # filename = "15GdH9-curt.mp3"

    use_vad = False
    all_outputs = process_file(filename, model, processor, transcribe_prompt, use_vad)

    # Combine and print results
    # print("==== FINAL RESULT ====")
    # for i, segment in enumerate(all_outputs):
    #    print(f"{segment}")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print total time used
    print(f"Vad: {use_vad}")
    print(f"Total time used: {elapsed_time:.2f} seconds")
    return all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Process an input file and output directory."
    )

    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to the output directory."
    )

    parser.add_argument("input_file", type=str, help="Path to the input file.")

    args = parser.parse_args()

    results = transcribe_file(args.input_file)

    output_path = os.path.join(
        args.output_dir, os.path.basename(args.input_file) + ".txt"
    )
    with open(output_path, "w") as outfile:
        for result in results:
            outfile.write(result)

    print(f"File copied to {output_path}")


if __name__ == "__main__":
    main()
