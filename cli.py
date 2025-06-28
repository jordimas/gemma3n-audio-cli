import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from file import process_file

GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

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

# Does not seem to honor JSON
transcribe_prompt = {
    "type": "text",
    "text": "Transcribe the audio file in Catalan with maximum accuracy\n"
}


filename = "dosparlants.mp3"
# filename = "15GdH9-curt.mp3"

all_outputs = process_file(filename, model, processor, transcribe_prompt)

# Combine and print results
print("==== FINAL RESULT ====")
for i, segment in enumerate(all_outputs):
    print(f"{segment}")
