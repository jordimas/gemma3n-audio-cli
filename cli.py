import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="")
model = AutoModelForImageTextToText.from_pretrained(
            GEMMA_MODEL_ID, torch_dtype="auto", device_map="cpu")
            
            
identify_lang_prompt  = {"type": "text", "text": "Identify the languages of the audio and generate a JSON with the following fields\n"
    "- language: a iso-639 language identified\n"
    "- confidence level from 0 to 1 of the language identified\n"
}

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "dosparlants.mp3"},
            identify_lang_prompt,
#            {"type": "text", "text": "Identify the speakers of this audio with names SPEAKER_01, SPEAKER_02, etc"},
#             {"type": "text", "text": "Identify the languages of the audio and the confidence level"},

        ]
    }
]
input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt",
)
input_ids = input_ids.to(model.device, dtype=model.dtype)

# Generate output from the model
outputs = model.generate(**input_ids, max_new_tokens=128)

# decode and print the output as text
text = processor.batch_decode(
    outputs,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print("result: " + text[0])

