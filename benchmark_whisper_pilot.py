import csv
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import whisper
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer


# Model is provided through the PowerShell command.
# Example: python benchmark_whisper_pilot.py small.en
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "small.en"

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "recording_script_pilot.csv"
)
AUDIO_DIR = (
    PROJECT_DIR
    / "audio"
    / "pilot"
    / "processed"
)
REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "whisper_benchmark"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

safe_model_name = MODEL_NAME.replace(".", "_")

RESULTS_PATH = (
    REPORTS_DIR
    / f"{safe_model_name}_pilot_results.csv"
)


# Select hardware.
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("WHISPER PILOT BENCHMARK")
print("=" * 60)
print("Model:", MODEL_NAME)
print("Device:", device)


# Load the selected model once.
load_start = time.perf_counter()

model = whisper.load_model(
    MODEL_NAME,
    device=device,
)

model_load_time = time.perf_counter() - load_start

print(f"Model loading time: {model_load_time:.2f} seconds")


# Load the 20 reference cases.
with DATASET_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    cases = list(csv.DictReader(file))

print("Loaded cases:", len(cases))

if len(cases) != 20:
    raise ValueError(
        f"Expected 20 cases, but loaded {len(cases)}."
    )


normalizer = EnglishTextNormalizer()

results = []
normalized_references = []
normalized_predictions = []

total_audio_duration = 0.0
total_inference_time = 0.0


if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Evaluate all 20 recordings.
for index, case in enumerate(cases, start=1):
    source_filename = case["filename"]

    wav_filename = (
        Path(source_filename).stem + ".wav"
    )

    audio_path = AUDIO_DIR / wav_filename
    reference = case["transcript"].strip()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Missing audio file: {audio_path}"
        )

    audio, sample_rate = sf.read(audio_path)

    if sample_rate != 16000:
        raise ValueError(
            f"{wav_filename}: expected 16000 Hz, "
            f"received {sample_rate} Hz."
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype("float32")
    audio_duration = len(audio) / sample_rate

    if device == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    transcription = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        verbose=False,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - inference_start
    )

    prediction = transcription["text"].strip()

    normalized_reference = normalizer(reference)
    normalized_prediction = normalizer(prediction)

    case_wer = wer(
        normalized_reference,
        normalized_prediction,
    )

    case_rtf = inference_time / audio_duration

    normalized_references.append(
        normalized_reference
    )
    normalized_predictions.append(
        normalized_prediction
    )

    total_audio_duration += audio_duration
    total_inference_time += inference_time

    results.append(
        {
            "filename": source_filename,
            "event_class": case["event_class"],
            "category": case["category"],
            "reference": reference,
            "prediction": prediction,
            "wer": round(case_wer, 4),
            "audio_duration_seconds": round(
                audio_duration,
                3,
            ),
            "inference_time_seconds": round(
                inference_time,
                3,
            ),
            "real_time_factor": round(
                case_rtf,
                4,
            ),
        }
    )

    print(
        f"[{index:02d}/20] "
        f"{source_filename} | "
        f"WER={case_wer:.4f} | "
        f"RTF={case_rtf:.4f}"
    )


# Aggregate evaluation metrics.
corpus_wer = wer(
    normalized_references,
    normalized_predictions,
)

mean_case_wer = (
    sum(result["wer"] for result in results)
    / len(results)
)

overall_rtf = (
    total_inference_time
    / total_audio_duration
)

if device == "cuda":
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )
else:
    peak_gpu_memory_gb = None


# Save detailed case results.
with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(results)
average_inference_time = (
    total_inference_time / len(results)
)

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print("Model:", MODEL_NAME)
print("Files evaluated:", len(results))
print(f"Corpus WER: {corpus_wer:.4f}")
print(f"Corpus WER percentage: {corpus_wer * 100:.2f}%")
print(f"Mean case WER: {mean_case_wer:.4f}")
print(f"Total audio duration: {total_audio_duration:.2f} seconds")
print(f"Total inference time: {total_inference_time:.2f} seconds")
print(
    f"Average inference time: "
    f"{average_inference_time:.2f} seconds"
)
print(f"Overall RTF: {overall_rtf:.4f}")
print(f"Model loading time: {model_load_time:.2f} seconds")

if peak_gpu_memory_gb is not None:
    print(
        f"Peak GPU memory: "
        f"{peak_gpu_memory_gb:.2f} GB"
    )

print("Results saved to:", RESULTS_PATH)