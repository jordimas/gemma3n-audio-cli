import os
import time
import json
from itertools import islice
from tqdm import tqdm
import soundfile as sf
import numpy as np
from datasets import load_dataset
from cli import transcribe_file, GEMMA_MODEL_ID, USE_VAD
from prompts import transcribe_prompt
from file import TEMPERATURE
from evaluate import load as load_metric


def main():
    print("Tool evaluation")

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    MAX_SAMPLES = 100
    datasets = ["mozilla-foundation/common_voice_16_1"]

    wer_metric = load_metric("wer")

    # Store WERs for each run per dataset
    all_wer_scores = {dataset: [] for dataset in datasets}
    all_stats = {dataset: [] for dataset in datasets}

    num_repeats = 4

    for repeat_i in range(num_repeats):
        print(f"\n--- Repeat {repeat_i + 1} / {num_repeats} ---")

        for dataset in datasets:
            dataset_iter = load_dataset(dataset, "en", split="test", streaming=True)
            dataset_iter = islice(dataset_iter, MAX_SAMPLES)

            predictions = []
            references = []

            start_time = time.time()

            for i, sample in enumerate(
                tqdm(dataset_iter, desc=f"Transcribing {dataset}")
            ):
                filename = f"audios/output_{i}.wav"
                audio = sample["audio"]
                chunk = audio["array"]
                sample_rate = audio["sampling_rate"]

                sf.write(filename, chunk, sample_rate)

                prediction, elapsed_time = transcribe_file(filename)
                prediction = str(prediction[0])

                reference = sample["sentence"]

                print(f"Compare '{prediction}' - '{reference}'")
                predictions.append(prediction)
                references.append(reference)

            total_time = time.time() - start_time
            sample_count = len(references)

            wer_score = (
                wer_metric.compute(predictions=predictions, references=references) * 100
            )

            all_wer_scores[dataset].append(wer_score)
            all_stats[dataset].append(
                {
                    "wer": round(wer_score, 2),
                    "total_time_sec": round(total_time, 2),
                    "avg_time_per_sample_sec": round(total_time / sample_count, 2),
                    "samples": sample_count,
                }
            )

            print(f"Samples processed: {sample_count}")
            print(f"WER for {dataset}: {wer_score:.2f}")
            print(f"Total inference time: {total_time:.2f} seconds")
            print(f"Average time per sample: {total_time / sample_count:.2f} seconds")

    # Compute average and std for WER per dataset
    lang_stats = {}
    for dataset in datasets:
        wer_array = np.array(all_wer_scores[dataset])
        wer = round(float(np.mean(wer_array)), 2)
        std = round(float(np.std(wer_array)), 2)
        lang_stats[dataset] = {
            "runs": all_stats[dataset],
            "wer_avg": wer,
            "wer_std": std,
        }
        print(f"WER avg for {dataset}: {wer_score:.2f}")
        print(f"WER std for {dataset}: {std:.2f}")

    # Add configuration info
    lang_stats["configuration"] = {
        "model": GEMMA_MODEL_ID,
        "use_vad": USE_VAD,
        "temperature": TEMPERATURE,
        "samples": MAX_SAMPLES,
        "prompt": transcribe_prompt,
    }

    with open("eval.json", "w") as f:
        json.dump(lang_stats, f, indent=2)


if __name__ == "__main__":
    main()
