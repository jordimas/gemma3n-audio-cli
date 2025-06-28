import torchaudio


def process_file(filename, model, processor, prompt):
    waveform, sample_rate = torchaudio.load(filename)
    t = waveform.shape[1] / sample_rate
    print(f"{filename} - {t:.2f} seconds")
    print(f"prompt: {prompt}")

    chunk_duration = 30  # seconds
    chunk_samples = int(chunk_duration * sample_rate)

    # Split into chunks
    chunks = [
        waveform[:, i : i + chunk_samples]
        for i in range(0, waveform.shape[1], chunk_samples)
    ]
    print(f"chunks: {len(chunks)}")

    all_outputs = []
    for i, chunk in enumerate(chunks):
        # Save temporary audio file
        temp_filename = f"chunk_{i}.wav"
        torchaudio.save(temp_filename, chunk, sample_rate)

        # Build message input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": temp_filename},
                    prompt,
                ],
            }
        ]

        # Convert chat to token IDs
        input_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(model.device, dtype=model.dtype)

        # Track input length for slicing generated output
        input_len = input_ids["input_ids"].shape[-1]

        # Generate response
        outputs = model.generate(
            **input_ids,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Extract only the generated part
        generated_ids = outputs.sequences[0][input_len:]
        answer = processor.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"processing: {answer}")
        all_outputs.append(answer)

    return all_outputs

