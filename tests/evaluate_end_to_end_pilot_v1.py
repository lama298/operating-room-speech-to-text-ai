import csv
import json
from pathlib import Path
from typing import Any

from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer

from src.end_to_end_pipeline_v1 import (
    EndToEndClinicalPipelineV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIRECTORY = (
    PROJECT_ROOT
    / "audio"
    / "pilot"
    / "raw"
)

RECORDING_SCRIPT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "recording_script_pilot.csv"
)

EXPECTED_OUTPUTS_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "expected_outputs_pilot.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "end_to_end_evaluation"
)

PROCESSED_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "processed_audio"
)

STRUCTURED_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "structured_records"
)

RESULTS_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "end_to_end_results.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "evaluation_summary.json"
)

TEXT_NORMALIZER = EnglishTextNormalizer()


def has_value(value: Any) -> bool:
    """
    Determine whether a value contains reportable data.

    False and zero are valid values.
    """

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return any(has_value(item) for item in value)

    if isinstance(value, dict):
        return any(
            has_value(item)
            for item in value.values()
        )

    return True


def flatten_schema(
    data: dict[str, Any],
    parent_path: str = "",
) -> dict[str, Any]:
    """
    Convert populated nested schema fields into
    dot-separated paths.

    Example:
    diagnosis_and_procedure.preoperative_diagnosis
    """

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
                flatten_schema(
                    value,
                    parent_path=path,
                )
            )

        elif has_value(value):
            flattened[path] = value

    return flattened


def get_nested_value(
    data: dict[str, Any],
    dotted_path: str,
) -> Any:
    """
    Retrieve a nested value using dot notation.
    """

    current: Any = data

    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None

        if part not in current:
            return None

        current = current[part]

    return current


def normalize_value(value: Any) -> Any:
    """
    Normalize values before comparison.
    """

    if isinstance(value, str):
        return TEXT_NORMALIZER(value)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, list):
        normalized_items = [
            normalize_value(item)
            for item in value
        ]

        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

    if isinstance(value, dict):
        return {
            key: normalize_value(item)
            for key, item in sorted(value.items())
        }

    return value


def values_equal(
    actual: Any,
    expected: Any,
) -> bool:
    """
    Compare actual and expected clinical values.
    """

    return normalize_value(actual) == normalize_value(
        expected
    )


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    """
    Calculate a safe percentage.
    """

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


def load_recording_script() -> list[dict[str, str]]:
    """
    Load reference transcripts and recording metadata.
    """

    with RECORDING_SCRIPT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_expected_outputs() -> dict[str, Any]:
    """
    Load Schema V2 ground-truth outputs.
    """

    with EXPECTED_OUTPUTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    recordings = load_recording_script()
    expected_outputs = load_expected_outputs()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    STRUCTURED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Pilot recordings:", len(recordings))
    print("Loading the end-to-end pipeline once...")

    pipeline = EndToEndClinicalPipelineV1(
        whisper_model_name="turbo",
        language="en",
    )

    results = []
    reference_transcripts = []
    predicted_transcripts = []

    total_true_positive_fields = 0
    total_false_positive_fields = 0
    total_false_negative_fields = 0
    total_expected_values = 0
    total_correct_values = 0

    try:
        for index, recording in enumerate(
            recordings,
            start=1,
        ):
            filename = recording["filename"]
            recording_id = Path(filename).stem

            if filename not in expected_outputs:
                raise ValueError(
                    f"Missing ground truth for {filename}"
                )

            ground_truth = expected_outputs[filename]

            input_audio_path = (
                AUDIO_DIRECTORY
                / filename
            )

            if not input_audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: "
                    f"{input_audio_path}"
                )

            processed_audio_path = (
                PROCESSED_DIRECTORY
                / f"{recording_id}.wav"
            )

            structured_output_path = (
                STRUCTURED_DIRECTORY
                / f"{recording_id}.json"
            )

            print(
                f"\n[{index}/{len(recordings)}] "
                f"Processing {filename}"
            )

            # record_id is intentionally omitted.
            # Evaluation must not populate MongoDB.
            pipeline_result = pipeline.process_audio(
                input_audio_path=input_audio_path,
                processed_audio_path=processed_audio_path,
                schema_output_path=structured_output_path,
                record_id=None,
                top_k=3,
            )

            reference_transcript = recording[
                "transcript"
            ]

            predicted_transcript = pipeline_result[
                "transcript"
            ]

            normalized_reference = TEXT_NORMALIZER(
                reference_transcript
            )

            normalized_prediction = TEXT_NORMALIZER(
                predicted_transcript
            )

            file_wer = wer(
                normalized_reference,
                normalized_prediction,
            )

            expected_schema = ground_truth.get(
                "expected_schema",
                {},
            )

            actual_schema = pipeline_result[
                "structured_record"
            ]

            expected_fields = flatten_schema(
                expected_schema
            )

            actual_fields = flatten_schema(
                actual_schema
            )

            expected_paths = set(
                expected_fields
            )

            actual_paths = set(
                actual_fields
            )

            true_positive_paths = (
                expected_paths & actual_paths
            )

            false_positive_paths = (
                actual_paths - expected_paths
            )

            false_negative_paths = (
                expected_paths - actual_paths
            )

            correct_value_paths = {
                path
                for path in true_positive_paths
                if values_equal(
                    actual_fields[path],
                    expected_fields[path],
                )
            }

            incorrect_value_paths = (
                true_positive_paths
                - correct_value_paths
            )

            expected_empty_fields = ground_truth.get(
                "expected_empty_fields",
                [],
            )

            empty_fields_correct = all(
                not has_value(
                    get_nested_value(
                        actual_schema,
                        path,
                    )
                )
                for path in expected_empty_fields
            )

            expected_persist = bool(
                ground_truth["persist"]
            )

            predicted_persist = (
                pipeline_result["mapping"][
                    "mapped_count"
                ] > 0
            )

            persistence_correct = (
                predicted_persist
                == expected_persist
            )

            field_precision = percentage(
                len(true_positive_paths),
                len(actual_paths),
            )

            field_recall = percentage(
                len(true_positive_paths),
                len(expected_paths),
            )

            if field_precision + field_recall:
                field_f1 = (
                    2
                    * field_precision
                    * field_recall
                    / (
                        field_precision
                        + field_recall
                    )
                )
            else:
                field_f1 = 0.0

            value_accuracy = percentage(
                len(correct_value_paths),
                len(expected_paths),
            )

            case_correct = (
                not false_positive_paths
                and not false_negative_paths
                and not incorrect_value_paths
                and empty_fields_correct
                and persistence_correct
            )

            total_true_positive_fields += len(
                true_positive_paths
            )

            total_false_positive_fields += len(
                false_positive_paths
            )

            total_false_negative_fields += len(
                false_negative_paths
            )

            total_expected_values += len(
                expected_paths
            )

            total_correct_values += len(
                correct_value_paths
            )

            reference_transcripts.append(
                normalized_reference
            )

            predicted_transcripts.append(
                normalized_prediction
            )

            row = {
                "filename": filename,
                "event_class": ground_truth[
                    "event_class"
                ],
                "reference_transcript": (
                    reference_transcript
                ),
                "predicted_transcript": (
                    predicted_transcript
                ),
                "wer": round(file_wer, 4),
                "expected_field_count": len(
                    expected_paths
                ),
                "predicted_field_count": len(
                    actual_paths
                ),
                "true_positive_fields": len(
                    true_positive_paths
                ),
                "false_positive_fields": len(
                    false_positive_paths
                ),
                "false_negative_fields": len(
                    false_negative_paths
                ),
                "field_precision_percent": round(
                    field_precision,
                    2,
                ),
                "field_recall_percent": round(
                    field_recall,
                    2,
                ),
                "field_f1_percent": round(
                    field_f1,
                    2,
                ),
                "correct_values": len(
                    correct_value_paths
                ),
                "incorrect_values": len(
                    incorrect_value_paths
                ),
                "value_accuracy_percent": round(
                    value_accuracy,
                    2,
                ),
                "expected_persist": expected_persist,
                "predicted_persist": (
                    predicted_persist
                ),
                "persistence_correct": (
                    persistence_correct
                ),
                "empty_fields_correct": (
                    empty_fields_correct
                ),
                "case_correct": case_correct,
                "expected_paths": json.dumps(
                    sorted(expected_paths)
                ),
                "actual_paths": json.dumps(
                    sorted(actual_paths)
                ),
                "false_positive_paths": json.dumps(
                    sorted(false_positive_paths)
                ),
                "false_negative_paths": json.dumps(
                    sorted(false_negative_paths)
                ),
                "incorrect_value_paths": json.dumps(
                    sorted(incorrect_value_paths)
                ),
            }

            results.append(row)

            print("Transcript:", predicted_transcript)
            print("WER:", f"{file_wer:.2%}")
            print(
                "Field TP/FP/FN:",
                len(true_positive_paths),
                len(false_positive_paths),
                len(false_negative_paths),
            )
            print(
                "Persistence correct:",
                persistence_correct,
            )
            print("Case correct:", case_correct)

    finally:
        pipeline.close()

    corpus_wer = wer(
        reference_transcripts,
        predicted_transcripts,
    )

    field_precision = percentage(
        total_true_positive_fields,
        (
            total_true_positive_fields
            + total_false_positive_fields
        ),
    )

    field_recall = percentage(
        total_true_positive_fields,
        (
            total_true_positive_fields
            + total_false_negative_fields
        ),
    )

    if field_precision + field_recall:
        field_f1 = (
            2
            * field_precision
            * field_recall
            / (
                field_precision
                + field_recall
            )
        )
    else:
        field_f1 = 0.0

    value_accuracy = percentage(
        total_correct_values,
        total_expected_values,
    )

    correct_cases = sum(
        row["case_correct"]
        for row in results
    )

    correct_persistence = sum(
        row["persistence_correct"]
        for row in results
    )

    non_clinical_results = [
        row
        for row in results
        if row["event_class"] == "non_clinical"
    ]

    correct_non_clinical_rejections = sum(
        not row["predicted_persist"]
        for row in non_clinical_results
    )

    summary = {
        "recording_count": len(results),
        "corpus_wer": round(corpus_wer, 4),
        "corpus_wer_percent": round(
            corpus_wer * 100,
            2,
        ),
        "field_true_positives": (
            total_true_positive_fields
        ),
        "field_false_positives": (
            total_false_positive_fields
        ),
        "field_false_negatives": (
            total_false_negative_fields
        ),
        "field_precision_percent": round(
            field_precision,
            2,
        ),
        "field_recall_percent": round(
            field_recall,
            2,
        ),
        "field_f1_percent": round(
            field_f1,
            2,
        ),
        "value_accuracy_percent": round(
            value_accuracy,
            2,
        ),
        "correct_end_to_end_cases": correct_cases,
        "end_to_end_case_accuracy_percent": round(
            percentage(
                correct_cases,
                len(results),
            ),
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
        "non_clinical_recordings": len(
            non_clinical_results
        ),
        "correct_non_clinical_rejections": (
            correct_non_clinical_rejections
        ),
        "non_clinical_rejection_accuracy_percent": (
            round(
                percentage(
                    correct_non_clinical_rejections,
                    len(non_clinical_results),
                ),
                2,
            )
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

    print("\nFINAL PILOT EVALUATION")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDetailed results:", RESULTS_CSV_PATH)
    print("Summary:", SUMMARY_JSON_PATH)


if __name__ == "__main__":
    main()