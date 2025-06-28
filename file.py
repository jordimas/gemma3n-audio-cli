import torchaudio


def process_file(filename, model, processor, prompt):

    waveform, sample_rate = torchaudio.load(filename)
    t = waveform.shape[1] / sample_rate
    print(f"{filename} - {t}")

    chunk_duration = 30  # seconds
    chunk_samples = chunk_duration * sample_rate

    # Split into chunks
    chunks = [
        waveform[:, i : i + chunk_samples]
        for i in range(0, waveform.shape[1], chunk_samples)
    ]
    print(f"chunks: {len(chunks)}")

    # Iterate over chunks
    all_outputs = []
    for i, chunk in enumerate(chunks):
        # Save temporary audio file for each chunk
        temp_filename = f"chunk_{i}.wav"
        torchaudio.save(temp_filename, chunk, sample_rate)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": temp_filename},
                    prompt,
                ],
            }
        ]

        input_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(model.device, dtype=model.dtype)

        outputs = model.generate(**input_ids, max_new_tokens=128)

        text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            return_full_text=False,  # <- only returns new tokens, i.e., model output
        )
        print(f"processing: {text[0]}")
        all_outputs.append(text[0])
    return all_outputs
