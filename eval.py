import time
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
import json
import os
from cli import transcribe_file
import soundfile as sf
from evaluate import load as load_metric


def main():
    print("Evaluation")

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    MAX_SAMPLES = 2
    datasets = ["mozilla-foundation/common_voice_16_1"]

    wer_metric = load_metric("wer")
    lang_stats = {}

    for dataset in datasets:

        dataset_iter = load_dataset(dataset, "en", split="test", streaming=True)
        dataset_iter = islice(dataset_iter, MAX_SAMPLES)
        predictions = []
        references = []

        start_time = time.time()

        for sample in tqdm(dataset_iter, desc="Transcribing"):
            #        print(f"sample: {sample}")
            filename = "output.wav"
            chunk = sample["audio_array"]
            sample_rate = sample["sampling_rate"]
            sf.write(filename, chunk, sample_rate)
            prediction = transcribe_file(filename)

            reference = sample["sentence"]
            print(f"Compare '{prediction}' - '{reference}'")
            predictions.append(prediction)
            references.append(reference)

        total_time = time.time() - start_time
        sample_count = len(references)
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        wer_score = wer_score * 100

        with open("results.txt", "w", encoding="utf-8") as f:
            for ref, pred in zip(references, predictions):
                f.write(f"{ref}\n{pred}\n\n")

        # Print individual language stats
        print(f"lens: {len(predictions)} - {len(references)}")
        print(f"WER for {dataset.upper()}: {wer_score:.4f}")
        print(
            f"Total inference time for {sample_count} samples: {total_time:.2f} seconds"
        )
        print(
            f"Average inference time per sample: {total_time / sample_count:.2f} seconds"
        )

    # Save stats
    lang_stats[dataset] = {
        "wer": round(wer_score, 4),
        "total_time_sec": round(total_time, 2),
        "avg_time_per_sample_sec": round(total_time / sample_count, 2),
        "samples": sample_count,
        #        "model": GEMMA_MODEL_ID,
    }

    # Output all stats as JSON
    print("\n=== Final Stats Summary ===")

    with open("lang_stats.json", "w") as f:
        json.dump(lang_stats, f, indent=2)


if __name__ == "__main__":
    main()
