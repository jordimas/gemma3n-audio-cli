import torchaudio
import torch
import torchaudio.transforms as T
import time

TEMPERATURE = 0.1


def transcribe_chunk(processor, model, prompt, chunk):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": chunk},
                prompt,
            ],
        }
    ]

    start_time = time.time()
    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device, dtype=model.dtype)
    input_len = input_ids["input_ids"].shape[-1]

    outputs = model.generate(
        **input_ids,
        max_new_tokens=256,
        return_dict_in_generate=True,
        output_scores=False,  # Don't compute per-token scores
        temperature=TEMPERATURE,
    )

    generated_ids = outputs.sequences[0][input_len:]
    answer = processor.decode(generated_ids, skip_special_tokens=True).strip()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"transcribe_chunk: {elapsed_time:.2f} seconds")

    return answer


# Requirements https://ai.google.dev/gemma/docs/capabilities/audio


def _get_waveform(filename):
    start_time = time.time()
    # Load audio
    waveform, sample_rate = torchaudio.load(filename)

    # Convert to mono if stereo (by averaging channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to float32 if needed (torchaudio usually returns float32 already, but we ensure it)
    waveform = waveform.to(torch.float32)

    # Scale int16 audio if needed (torchaudio.load handles scaling, but add check if using raw decoding)
    if waveform.max() > 1.0 or waveform.min() < -1.0:
        # Assume raw int16 range [-32768, 32767]
        waveform /= 32768.0

    # Resample to 16kHz if needed
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
        print("Resampled to 16 kHz")

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"_get_waveform: {elapsed_time:.2f} seconds")

    print(f"{filename} - {waveform.shape[1] / sample_rate:.2f} seconds")
    return waveform, sample_rate


vad_model = None
utils = None


def process_file(filename, model, processor, prompt, use_vad=False):
    global vad_model, utils

    waveform, sample_rate = _get_waveform(filename)

    if use_vad:
        # torch.set_num_threads(1)
        if not vad_model:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
        (get_speech_timestamps, _, _, _, _) = utils

        # Apply VAD to get speech timestamps
        speech_timestamps = get_speech_timestamps(
            waveform, vad_model, sampling_rate=sample_rate
        )

        if not speech_timestamps:
            print("Warning: No speech detected by VAD.")
        else:
            # waveform = torch.cat(
            #    [waveform[:, ts["start"] : ts["end"]] for ts in speech_timestamps], dim=1
            # )
            end = speech_timestamps[-1]["end"]

            # Slice the waveform to keep only from start to end
            waveform = waveform[:, 0:end]

            print(
                f"VAD-reduced duration: {waveform.shape[1] / sample_rate:.2f} seconds"
            )

    chunk_duration = 30  # seconds
    chunk_samples = int(chunk_duration * sample_rate)

    # Split into chunks
    chunks = [
        waveform[:, i : i + chunk_samples]
        for i in range(0, waveform.shape[1], chunk_samples)
    ]
    if len(chunks) > 1:
        print(f"chunks: {len(chunks)}")

    all_outputs = []
    for i, chunk in enumerate(chunks):
        #        temp_filename = f"chunk_{i}.wav"
        temp_filename = "chunk.wav"
        torchaudio.save(temp_filename, chunk, sample_rate)

        answer = transcribe_chunk(processor, model, prompt, temp_filename)
        #        print(f"{answer}")
        all_outputs.append(answer)

    return all_outputs
