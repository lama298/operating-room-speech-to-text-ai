"""Evaluate the frozen end-to-end pipeline on the 30-case final-test split."""

import csv
import json
from pathlib import Path
from typing import Any

from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer

from src.end_to_end_pipeline_v1 import EndToEndClinicalPipelineV1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIRECTORY = PROJECT_ROOT / "audio" / "final_test" / "raw"
RECORDING_SCRIPT_PATH = PROJECT_ROOT / "dataset" / "recording_script_final_test.csv"
EXPECTED_OUTPUTS_PATH = PROJECT_ROOT / "dataset" / "expected_outputs_final_test.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "final_test_evaluation"
PROCESSED_DIRECTORY = OUTPUT_DIRECTORY / "processed_audio"
STRUCTURED_DIRECTORY = OUTPUT_DIRECTORY / "structured_records"
RESULTS_CSV_PATH = OUTPUT_DIRECTORY / "final_test_results.csv"
SUMMARY_JSON_PATH = OUTPUT_DIRECTORY / "evaluation_summary.json"
TEXT_NORMALIZER = EnglishTextNormalizer()


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_value(item) for item in value)
    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())
    return True


def flatten_schema(data: dict[str, Any], parent_path: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        if key in {"_id", "schema_version", "processing_metadata"}:
            continue
        path = f"{parent_path}.{key}" if parent_path else key
        if isinstance(value, dict):
            flattened.update(flatten_schema(value, path))
        elif has_value(value):
            flattened[path] = value
    return flattened


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return TEXT_NORMALIZER(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        items = [normalize_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in sorted(value.items())}
    return value


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> None:
    with RECORDING_SCRIPT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        recordings = list(csv.DictReader(file))
    with EXPECTED_OUTPUTS_PATH.open("r", encoding="utf-8") as file:
        expected_outputs = json.load(file)

    filenames = [row["filename"] for row in recordings]
    if len(recordings) != 30 or len(set(filenames)) != 30:
        raise ValueError("Final-test manifest must contain 30 unique recordings.")
    if set(filenames) != set(expected_outputs):
        raise ValueError("Final-test manifest and expected-output IDs do not match.")

    for directory in (OUTPUT_DIRECTORY, PROCESSED_DIRECTORY, STRUCTURED_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline = EndToEndClinicalPipelineV1(whisper_model_name="turbo", language="en")
    results = []
    reference_transcripts = []
    predicted_transcripts = []
    total_tp = total_fp = total_fn = total_expected = total_correct_values = 0

    try:
        for index, recording in enumerate(recordings, start=1):
            filename = recording["filename"]
            recording_id = Path(filename).stem
            input_path = AUDIO_DIRECTORY / filename
            if not input_path.exists():
                raise FileNotFoundError(f"Audio file not found: {input_path}")

            print(f"\n[{index}/{len(recordings)}] Processing {filename}")
            result = pipeline.process_audio(
                input_audio_path=input_path,
                processed_audio_path=PROCESSED_DIRECTORY / f"{recording_id}.wav",
                schema_output_path=STRUCTURED_DIRECTORY / f"{recording_id}.json",
                record_id=None,
                top_k=3,
            )

            truth = expected_outputs[filename]
            reference = recording["transcript"]
            prediction = result["transcript"]
            normalized_reference = TEXT_NORMALIZER(reference)
            normalized_prediction = TEXT_NORMALIZER(prediction)
            file_wer = wer(normalized_reference, normalized_prediction)
            reference_transcripts.append(normalized_reference)
            predicted_transcripts.append(normalized_prediction)

            expected_fields = flatten_schema(truth.get("expected_schema", {}))
            actual_fields = flatten_schema(result["structured_record"])
            expected_paths = set(expected_fields)
            actual_paths = set(actual_fields)
            tp_paths = expected_paths & actual_paths
            fp_paths = actual_paths - expected_paths
            fn_paths = expected_paths - actual_paths
            correct_value_paths = {
                path for path in tp_paths
                if normalize_value(actual_fields[path]) == normalize_value(expected_fields[path])
            }
            incorrect_value_paths = tp_paths - correct_value_paths
            expected_persist = bool(truth["persist"])
            predicted_persist = result["mapping"]["mapped_count"] > 0
            persistence_correct = predicted_persist == expected_persist
            case_correct = not fp_paths and not fn_paths and not incorrect_value_paths and persistence_correct

            total_tp += len(tp_paths)
            total_fp += len(fp_paths)
            total_fn += len(fn_paths)
            total_expected += len(expected_paths)
            total_correct_values += len(correct_value_paths)

            row = {
                "filename": filename,
                "event_class": truth["event_class"],
                "reference_transcript": reference,
                "predicted_transcript": prediction,
                "wer": round(file_wer, 4),
                "expected_field_count": len(expected_paths),
                "predicted_field_count": len(actual_paths),
                "true_positive_fields": len(tp_paths),
                "false_positive_fields": len(fp_paths),
                "false_negative_fields": len(fn_paths),
                "correct_values": len(correct_value_paths),
                "incorrect_values": len(incorrect_value_paths),
                "expected_persist": expected_persist,
                "predicted_persist": predicted_persist,
                "persistence_correct": persistence_correct,
                "case_correct": case_correct,
                "expected_paths": json.dumps(sorted(expected_paths)),
                "actual_paths": json.dumps(sorted(actual_paths)),
                "false_positive_paths": json.dumps(sorted(fp_paths)),
                "false_negative_paths": json.dumps(sorted(fn_paths)),
                "incorrect_value_paths": json.dumps(sorted(incorrect_value_paths)),
            }
            results.append(row)
            print("Transcript:", prediction)
            print("WER:", f"{file_wer:.2%}")
            print("Field TP/FP/FN:", len(tp_paths), len(fp_paths), len(fn_paths))
            print("Persistence correct:", persistence_correct)
            print("Case correct:", case_correct)
    finally:
        pipeline.close()

    precision = percentage(total_tp, total_tp + total_fp)
    recall = percentage(total_tp, total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    non_clinical = [row for row in results if row["event_class"] == "non_clinical"]
    summary = {
        "split": "final_test",
        "recording_count": len(results),
        "corpus_wer": round(wer(reference_transcripts, predicted_transcripts), 4),
        "corpus_wer_percent": round(wer(reference_transcripts, predicted_transcripts) * 100, 2),
        "field_true_positives": total_tp,
        "field_false_positives": total_fp,
        "field_false_negatives": total_fn,
        "field_precision_percent": round(precision, 2),
        "field_recall_percent": round(recall, 2),
        "field_f1_percent": round(f1, 2),
        "value_accuracy_percent": round(percentage(total_correct_values, total_expected), 2),
        "correct_end_to_end_cases": sum(row["case_correct"] for row in results),
        "end_to_end_case_accuracy_percent": round(percentage(sum(row["case_correct"] for row in results), len(results)), 2),
        "correct_persistence_decisions": sum(row["persistence_correct"] for row in results),
        "persistence_accuracy_percent": round(percentage(sum(row["persistence_correct"] for row in results), len(results)), 2),
        "non_clinical_recordings": len(non_clinical),
        "correct_non_clinical_rejections": sum(not row["predicted_persist"] for row in non_clinical),
        "non_clinical_rejection_accuracy_percent": round(percentage(sum(not row["predicted_persist"] for row in non_clinical), len(non_clinical)), 2),
    }

    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\nFINAL TEST EVALUATION")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nDetailed results:", RESULTS_CSV_PATH)
    print("Summary:", SUMMARY_JSON_PATH)


if __name__ == "__main__":
    main()
