import torchaudio
import torch
import numpy as np
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
import torchaudio.transforms as T

print(f"threads: {torch.get_num_threads()}")
torch.set_num_threads(12) 
print(f"threads: {torch.get_num_threads()}")

# Load Silero VAD model
torch.set_num_threads(1)  # Optional: improves performance in some environments
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, _, _, _) = utils


def process_file(filename, model, processor, prompt, use_vad=False):
    waveform, sample_rate = torchaudio.load(filename)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    print(f"{filename} - {waveform.shape[1] / sample_rate:.2f} seconds")
    print(f"prompt: {prompt}")

    # Resample if not 16000 Hz
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    if use_vad:
        # Apply VAD to get speech timestamps
        speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)

        # Concatenate speech segments
        waveform = torch.cat([
            waveform[:, ts['start']:ts['end']] for ts in speech_timestamps
        ], dim=1)

        print(f"VAD-reduced duration: {waveform.shape[1] / sample_rate:.2f} seconds")

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
        input_len = input_ids["input_ids"].shape[-1]

        outputs = model.generate(
            **input_ids,
            max_new_tokens=8192,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][input_len:]
        answer = processor.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"{answer}")
        all_outputs.append(answer)

    return all_outputs

