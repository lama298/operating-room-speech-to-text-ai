"""Evaluate the frozen pipeline on the 20-case fresh blind-test split."""

import csv
import json
from pathlib import Path
from typing import Any

import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIRECTORY = (
    PROJECT_ROOT / "audio" / "test" / "raw"
)

RECORDING_SCRIPT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "test"
    / "recording_plan.csv"
)

EXPECTED_OUTPUTS_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "test"
    / "expected_outputs_fresh_blind.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "fresh_blind_post_improvement_regression"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT / "audio" / "test" / "processed"
)

STRUCTURED_DIRECTORY = (
    OUTPUT_DIRECTORY / "structured_records"
)

RESULTS_CSV_PATH = (
    OUTPUT_DIRECTORY / "fresh_blind_test_results.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIRECTORY / "evaluation_summary.json"
)

TEXT_NORMALIZER = None


def has_value(value: Any) -> bool:
    """Return whether a Schema V2 value is populated."""

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return any(has_value(item) for item in value)

    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())

    return True


def flatten_schema(
    data: dict[str, Any],
    parent_path: str = "",
) -> dict[str, Any]:
    """Flatten populated Schema V2 leaves into dotted paths."""

    flattened = {}

    for key, value in data.items():
        if key in {
            "_id",
            "schema_version",
            "processing_metadata",
        }:
            continue

        path = (
            f"{parent_path}.{key}"
            if parent_path
            else key
        )

        if isinstance(value, dict):
            flattened.update(
                flatten_schema(value, path)
            )
        elif has_value(value):
            flattened[path] = value

    return flattened


def normalize_value(value: Any) -> Any:
    """Normalize values before exact ground-truth comparison."""

    if isinstance(value, str):
        return TEXT_NORMALIZER(value)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, list):
        items = [
            normalize_value(item)
            for item in value
        ]

        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
            ),
        )

    if isinstance(value, dict):
        return {
            key: normalize_value(item)
            for key, item in sorted(value.items())
        }

    return value


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    """Return a percentage while safely handling zero."""

    if not denominator:
        return 0.0

    return 100.0 * numerator / denominator


def main() -> None:
    """Run the complete frozen blind-test evaluation once."""

    parser = argparse.ArgumentParser(description="Evaluate the 20 fresh recordings.")
    parser.add_argument("--check-only", action="store_true", help="Check files without model inference.")
    args = parser.parse_args()

    with RECORDING_SCRIPT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        recordings = list(csv.DictReader(file))

    with EXPECTED_OUTPUTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        expected_outputs = json.load(file)

    filenames = [
        row["filename"]
        for row in recordings
    ]

    if (
        len(recordings) != 20
        or len(set(filenames)) != 20
    ):
        raise ValueError(
            "Blind-test manifest must contain "
            "20 unique recordings."
        )

    if set(filenames) != set(expected_outputs):
        raise ValueError(
            "Blind-test manifest and expected-output "
            "IDs do not match."
        )

    # Resolve recorder extensions by stem without changing the reference text.
    # Reject ambiguous matches instead of guessing which take to evaluate.
    allowed = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".mp4"}
    audio_files = [p for p in AUDIO_DIRECTORY.iterdir()
                   if p.is_file() and p.suffix.lower() in allowed]
    resolved_truth = {}
    used_files = set()
    for recording in recordings:
        original_name = recording["filename"]
        matches = [p for p in audio_files
                   if p.stem.casefold() == Path(original_name).stem.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Expected one audio file for {original_name}; found "
                             f"{len(matches)}. Check names and duplicate takes.")
        actual_name = matches[0].name
        used_files.add(actual_name)
        recording["filename"] = actual_name
        resolved_truth[actual_name] = expected_outputs[original_name]
        if actual_name != original_name:
            print(f"Filename match: {original_name} -> {actual_name}")
    expected_outputs = resolved_truth
    extras = sorted(p.name for p in audio_files if p.name not in used_files)
    if extras:
        raise ValueError(f"Unlisted audio files in fresh-test folder: {extras}. "
                         "Move unrelated files to a different folder.")
    for recording in recordings:
        truth = expected_outputs[recording["filename"]]
        if not recording.get("script", "").strip():
            raise ValueError(f"Empty reference transcript: {recording['filename']}")
        if not isinstance(truth.get("persist"), bool):
            raise ValueError("Each persist label must be a JSON boolean.")
        if truth["event_class"] != recording["event_class"]:
            raise ValueError("Manifest and ground-truth event classes differ.")
    print(f"Preflight passed: {len(recordings)} recordings and matching ground truth.")
    print("Audio folder:", AUDIO_DIRECTORY)
    print("Results folder:", OUTPUT_DIRECTORY)
    if args.check_only:
        print("Check only: no models loaded and no inference performed.")
        return

    # Load the existing pipeline only after all dataset checks have passed.
    from jiwer import wer
    from whisper.normalizers import EnglishTextNormalizer
    from src.end_to_end_pipeline_v1 import EndToEndClinicalPipelineV1
    global TEXT_NORMALIZER
    TEXT_NORMALIZER = EnglishTextNormalizer()

    for directory in (
        OUTPUT_DIRECTORY,
        PROCESSED_DIRECTORY,
        STRUCTURED_DIRECTORY,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Retain the exact filename resolution used in this run for auditability.
    with (OUTPUT_DIRECTORY / "effective_recording_plan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=recordings[0].keys())
        writer.writeheader()
        writer.writerows(recordings)
    with (OUTPUT_DIRECTORY / "effective_expected_outputs.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(expected_outputs, file, indent=2, ensure_ascii=False)

    pipeline = EndToEndClinicalPipelineV1(
        whisper_model_name="turbo",
        language="en",
    )

    results = []
    reference_transcripts = []
    predicted_transcripts = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_expected = 0
    total_correct_values = 0

    try:
        for index, recording in enumerate(
            recordings,
            start=1,
        ):
            filename = recording["filename"]
            recording_id = Path(filename).stem
            input_path = AUDIO_DIRECTORY / filename

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {input_path}"
                )

            print(
                f"\n[{index}/{len(recordings)}] "
                f"Processing {filename}"
            )

            result = pipeline.process_audio(
                input_audio_path=input_path,
                processed_audio_path=(
                    PROCESSED_DIRECTORY
                    / f"{recording_id}.wav"
                ),
                schema_output_path=(
                    STRUCTURED_DIRECTORY
                    / f"{recording_id}.json"
                ),
                record_id=None,
                top_k=3,
            )

            truth = expected_outputs[filename]
            reference = recording["script"]
            prediction = result["transcript"]

            normalized_reference = (
                TEXT_NORMALIZER(reference)
            )
            normalized_prediction = (
                TEXT_NORMALIZER(prediction)
            )

            file_wer = wer(
                normalized_reference,
                normalized_prediction,
            )

            reference_transcripts.append(
                normalized_reference
            )
            predicted_transcripts.append(
                normalized_prediction
            )

            expected_fields = flatten_schema(
                truth.get("expected_schema", {})
            )
            actual_fields = flatten_schema(
                result["structured_record"]
            )

            expected_paths = set(expected_fields)
            actual_paths = set(actual_fields)
            tp_paths = expected_paths & actual_paths
            fp_paths = actual_paths - expected_paths
            fn_paths = expected_paths - actual_paths

            correct_value_paths = {
                path
                for path in tp_paths
                if normalize_value(actual_fields[path])
                == normalize_value(expected_fields[path])
            }

            incorrect_value_paths = (
                tp_paths - correct_value_paths
            )

            expected_persist = bool(truth["persist"])
            predicted_persist = (
                result["mapping"]["mapped_count"] > 0
            )
            persistence_correct = (
                predicted_persist == expected_persist
            )

            case_correct = (
                not fp_paths
                and not fn_paths
                and not incorrect_value_paths
                and persistence_correct
            )

            total_tp += len(tp_paths)
            total_fp += len(fp_paths)
            total_fn += len(fn_paths)
            total_expected += len(expected_paths)
            total_correct_values += len(
                correct_value_paths
            )

            row = {
                "filename": filename,
                "event_class": truth["event_class"],
                "reference_transcript": reference,
                "predicted_transcript": prediction,
                "wer": round(file_wer, 4),
                "expected_field_count": len(
                    expected_paths
                ),
                "predicted_field_count": len(
                    actual_paths
                ),
                "true_positive_fields": len(tp_paths),
                "false_positive_fields": len(fp_paths),
                "false_negative_fields": len(fn_paths),
                "correct_values": len(
                    correct_value_paths
                ),
                "incorrect_values": len(
                    incorrect_value_paths
                ),
                "expected_persist": expected_persist,
                "predicted_persist": predicted_persist,
                "persistence_correct": (
                    persistence_correct
                ),
                "case_correct": case_correct,
                "expected_paths": json.dumps(
                    sorted(expected_paths)
                ),
                "actual_paths": json.dumps(
                    sorted(actual_paths)
                ),
                "false_positive_paths": json.dumps(
                    sorted(fp_paths)
                ),
                "false_negative_paths": json.dumps(
                    sorted(fn_paths)
                ),
                "incorrect_value_paths": json.dumps(
                    sorted(incorrect_value_paths)
                ),
            }

            results.append(row)

            print("Transcript:", prediction)
            print("WER:", f"{file_wer:.2%}")
            print(
                "Field TP/FP/FN:",
                len(tp_paths),
                len(fp_paths),
                len(fn_paths),
            )
            print(
                "Persistence correct:",
                persistence_correct,
            )
            print("Case correct:", case_correct)

    finally:
        pipeline.close()

    precision = percentage(
        total_tp,
        total_tp + total_fp,
    )
    recall = percentage(
        total_tp,
        total_tp + total_fn,
    )

    if precision + recall:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    non_clinical = [
        row
        for row in results
        if row["event_class"] == "non_clinical"
    ]

    correct_cases = sum(
        row["case_correct"]
        for row in results
    )
    correct_persistence = sum(
        row["persistence_correct"]
        for row in results
    )
    correct_non_clinical = sum(
        not row["predicted_persist"]
        for row in non_clinical
    )

    corpus_wer = wer(
        reference_transcripts,
        predicted_transcripts,
    )

    summary = {
        "split": "fresh_blind_post_improvement_regression",
        "recording_count": len(results),
        "corpus_wer": round(corpus_wer, 4),
        "corpus_wer_percent": round(
            corpus_wer * 100,
            2,
        ),
        "field_true_positives": total_tp,
        "field_false_positives": total_fp,
        "field_false_negatives": total_fn,
        "field_precision_percent": round(
            precision,
            2,
        ),
        "field_recall_percent": round(recall, 2),
        "field_f1_percent": round(f1, 2),
        "value_accuracy_percent": round(
            percentage(
                total_correct_values,
                total_expected,
            ),
            2,
        ),
        "correct_end_to_end_cases": correct_cases,
        "end_to_end_case_accuracy_percent": round(
            percentage(correct_cases, len(results)),
            2,
        ),
        "correct_persistence_decisions": (
            correct_persistence
        ),
        "persistence_accuracy_percent": round(
            percentage(
                correct_persistence,
                len(results),
            ),
            2,
        ),
        "non_clinical_recordings": len(non_clinical),
        "correct_non_clinical_rejections": (
            correct_non_clinical
        ),
        "non_clinical_rejection_accuracy_percent": round(
            percentage(
                correct_non_clinical,
                len(non_clinical),
            ),
            2,
        ),
    }

    with RESULTS_CSV_PATH.open(
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

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nPOST-IMPROVEMENT REGRESSION EVALUATION")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDetailed results:", RESULTS_CSV_PATH)
    print("Summary:", SUMMARY_JSON_PATH)


if __name__ == "__main__":
    main()
