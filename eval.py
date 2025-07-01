import time
import os
import json
import soundfile as sf
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset
from evaluate import load as load_metric
from cli import transcribe_file, GEMMA_MODEL_ID, USE_VAD
from prompts import transcribe_prompt
from file import TEMPERATURE


def main():
    print("Tool evaluation")

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    MAX_SAMPLES = 200
    datasets = ["mozilla-foundation/common_voice_16_1"]

    wer_metric = load_metric("wer")
    lang_stats = {}

    for dataset in datasets:
        dataset_iter = load_dataset(dataset, "en", split="test", streaming=True)
        dataset_iter = islice(dataset_iter, MAX_SAMPLES)
        #        dataset_obj = load_dataset(dataset, "en", split="test")[:MAX_SAMPLES]
        #        print(f"Loaded {len(dataset_obj)} samples")
        #        print(dataset_obj.keys())/7
        #        dataset_iter = dataset_obj.select(range(min(MAX_SAMPLES, len(dataset_obj))))

        predictions = []
        references = []

        start_time = time.time()

        for i, sample in enumerate(tqdm(dataset_iter, desc=f"Transcribing {dataset}")):
            #            print(f"sample: {sample}" )
            filename = f"audios/output_{i}.wav"
            audio = sample["audio"]
            chunk = audio["array"]
            sample_rate = audio["sampling_rate"]

            # Write audio chunk to file
            sf.write(filename, chunk, sample_rate)

            # Transcribe
            prediction, elapsed_time = transcribe_file(filename)
            prediction = str(prediction[0])

            # Ground truth
            reference = sample["sentence"]
            print(f"Compare '{prediction}' - '{reference}'")

            predictions.append(prediction)
            references.append(reference)

        total_time = time.time() - start_time
        sample_count = len(references)

        # Compute WER
        #        print(f"references: {references}")
        #        print(f"predictions: {predictions}")

        wer_score = wer_metric.compute(predictions=predictions, references=references)
        wer_score = wer_score * 100

        # Write results
        with open("results.txt", "w", encoding="utf-8") as f:
            for ref, pred in zip(references, predictions):
                f.write(f"{ref}\n{pred}\n\n")

        # Print stats
        print(f"Samples processed: {sample_count}")
        print(f"WER for {dataset}: {wer_score:.2f}")
        print(f"Total inference time: {total_time:.2f} seconds")
        print(f"Average time per sample: {total_time / sample_count:.2f} seconds")

        # Save dataset-level stats
        lang_stats[dataset] = {
            "wer": round(wer_score, 2),
            "total_time_sec": round(total_time, 2),
            "avg_time_per_sample_sec": round(total_time / sample_count, 2),
            "samples": sample_count,
        }

    lang_stats["configuration"] = {
        "model": GEMMA_MODEL_ID,
        "use_vad": USE_VAD,
        "temperature": TEMPERATURE,
        "prompt": transcribe_prompt,
    }

    with open("eval.json", "w") as f:
        json.dump(lang_stats, f, indent=2)


if __name__ == "__main__":
    main()
