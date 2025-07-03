import os
import time
import json
import numpy as np
import soundfile as sf
from itertools import islice
from datasets import load_dataset
from cli import transcribe_file, load_model_and_processor
from evaluate import load as load_metric
from prompts import transcribe_prompt


def main():
    print("Tool evaluation")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

    MAX_SAMPLES = 50
    MODEL_NAME = "google/gemma-3n-E4B-it"
    TEMPERATURE = 0.1
    USE_VAD = False

    datasets = ["mozilla-foundation/common_voice_16_1"]
    wer_metric = load_metric("wer")

    all_wer_scores = {dataset: [] for dataset in datasets}
    all_stats = {dataset: [] for dataset in datasets}
    num_repeats = 1

    model, processor = load_model_and_processor(MODEL_NAME)

    for repeat_i in range(num_repeats):
        print(f"\n--- Repeat {repeat_i + 1} / {num_repeats} ---")

        for dataset in datasets:
            dataset_iter = load_dataset(dataset, "en", split="test", streaming=True)
            dataset_iter = islice(dataset_iter, MAX_SAMPLES)

            predictions = []
            references = []
            start_time = time.time()

            for i, sample in enumerate(dataset_iter):
                filename = f"audios/output_{i}.wav"
                audio = sample["audio"]
                sf.write(filename, audio["array"], audio["sampling_rate"])

                outputs, _ = transcribe_file(
                    filename, model, processor, transcribe_prompt, TEMPERATURE, USE_VAD
                )
                prediction = str(outputs[0])
                reference = sample["sentence"]

                print(f"Compare '{prediction}' - '{reference}'")
                predictions.append(prediction)
                references.append(reference)

            total_time = time.time() - start_time
            sample_count = len(references)

            wer_score = (
                wer_metric.compute(predictions=predictions, references=references) * 100
            )

            with open("results.txt", "w", encoding="utf-8") as f:
                for ref, pred in zip(references, predictions):
                    f.write(f"{ref}\n{pred}\n\n")

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

    # Summary
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

    lang_stats["configuration"] = {
        "model": MODEL_NAME,
        "use_vad": USE_VAD,
        "temperature": TEMPERATURE,
        "samples": MAX_SAMPLES,
        "prompt": transcribe_prompt,
    }

    with open("eval.json", "w") as f:
        json.dump(lang_stats, f, indent=2)


if __name__ == "__main__":
    main()
