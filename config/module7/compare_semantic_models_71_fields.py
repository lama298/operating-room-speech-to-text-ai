import csv
import gc #used to remove model from memory
import json
import time #measures loading and inference time
from pathlib import Path

import torch

from semantic_detector_v2 import (
    SemanticFieldDetectorV2,
) #use eexisting detector model
"""
Load validation statements
→ Load Model 1
→ Predict candidate fields and similarity scores
→ Find Model 1’s best threshold
→ Calculate accuracy, macro F1, and speed
→ Repeat for Models 2 and 3
→ Save and print the comparison
"""

MODULE_DIR = Path(__file__).resolve().parent #module 7
PROJECT_ROOT = MODULE_DIR.parent.parent #whisper thesis

VALIDATION_PATH = (
    MODULE_DIR
    / "semantic_validation_full_v2.json"
)

DETAILS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "semantic_model_details_71_fields.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "semantic_model_comparison_71_fields.csv"
)


MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "BioLORD": "FremyCompany/BioLORD-2023-C",
    "PubMedBERT": "NeuML/pubmedbert-base-embeddings",
    "BGE":"BAAI/bge-base-en-v1.5"

}


# Test thresholds from 0.00 to 0.80., each model gets its own threeshold
THRESHOLDS = [
    threshold / 100
    for threshold in range(0, 81)
]


def synchronize_cuda():
    """Wait for pending GPU operations before measuring time."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calculate_macro_f1(results, field_names): #evaluates how well the model distinguishes the 12 fields
    """Calculate candidate macro F1 over relevant statements."""
    #this function calculates one F1 score for each of the 12 fields

    field_f1_scores = []

    for field_name in field_names:
        true_positive = sum(
            result["expected_field"] == field_name
            and result["candidate_field"] == field_name
            for result in results
        )

        false_positive = sum(
            result["expected_field"] != field_name
            and result["candidate_field"] == field_name
            for result in results
            if result["expected_field"] is not None #excluding unrelateed statements so it only measures clinical field classification
        )

        false_negative = sum(
            result["expected_field"] == field_name
            and result["candidate_field"] != field_name
            for result in results
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall

        field_f1 = (
            2 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        field_f1_scores.append(field_f1)

    return sum(field_f1_scores) / len(field_f1_scores)


def evaluate_threshold(results, threshold):
    """Evaluate one threshold using all validation statements."""

    correct_clinical = 0
    rejected_clinical = 0
    accepted_wrong = 0
    rejected_unrelated = 0
    accepted_unrelated = 0

    for result in results:
        expected_field = result["expected_field"]
        candidate_field = result["candidate_field"]
        score = result["score"]

        accepted = score >= threshold

        if expected_field is not None: #it means statement is clinical
            candidate_correct = candidate_field == expected_field

            if accepted and candidate_correct:
                correct_clinical += 1
            elif accepted and not candidate_correct:
                accepted_wrong += 1
            else:
                rejected_clinical += 1

        else: #else statement is unrelated
            if accepted:
                accepted_unrelated += 1
            else:
                rejected_unrelated += 1

    total_correct = correct_clinical + rejected_unrelated #threshold accuracy
    accuracy = total_correct / len(results)

    return {
        "threshold": threshold,
        "correct_clinical": correct_clinical,
        "rejected_clinical": rejected_clinical,
        "accepted_wrong": accepted_wrong,
        "rejected_unrelated": rejected_unrelated,
        "accepted_unrelated": accepted_unrelated,
        "threshold_accuracy": accuracy,
    }


def select_best_threshold(results):
    """Choose the safest threshold among the best-performing ones."""

    threshold_results = [ #first evaluate 81 thresholds
        evaluate_threshold(results, threshold)
        for threshold in THRESHOLDS
    ]

    return max(
        threshold_results,
        key=lambda result: (
            result["threshold_accuracy"],
            -result["accepted_unrelated"],
            -result["accepted_wrong"],
            result["correct_clinical"],
            result["threshold"],
        ),
    )


with VALIDATION_PATH.open("r", encoding="utf-8") as file:
    validation_data = json.load(file)


all_details = []
model_summaries = []


for model_label, model_name in MODELS.items():
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_label}")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}")

    synchronize_cuda()
    setup_start = time.perf_counter()

    # threshold=0.0 prevents the detector's current default threshold
    # from affecting the comparison. Each model is calibrated below.
    detector = SemanticFieldDetectorV2(
        model_name=model_name,
        threshold=0.0,
    )

    synchronize_cuda()
    setup_seconds = time.perf_counter() - setup_start

    # Warm up the model before measuring prediction time.
    detector.predict(
        validation_data[0]["text"],
        top_k=3,
    )

    model_results = []
    inference_times = []

    for test_number, example in enumerate(
        validation_data,
        start=1,
    ):
        synchronize_cuda()
        prediction_start = time.perf_counter()

        prediction = detector.predict(
            example["text"],
            top_k=3,
        )

        synchronize_cuda()
        inference_seconds = (
            time.perf_counter() - prediction_start
        )

        inference_times.append(inference_seconds)

        result = {
            "test_number": test_number,
            "text": example["text"],
            "expected_field": example["expected_field"],
            "candidate_field": prediction["candidate_field"],
            "score": prediction["score"],
        }

        model_results.append(result)

        top_predictions = prediction["top_predictions"]

        all_details.append(
            {
                "model_label": model_label,
                "model_name": model_name,
                "test_number": test_number,
                "text": example["text"],
                "expected_field": example["expected_field"],
                "candidate_field": prediction["candidate_field"],
                "score": prediction["score"],
                "matched_reference": (
                    prediction["matched_reference"]
                ),
                "top_1_field": top_predictions[0]["field"],
                "top_1_score": top_predictions[0]["score"],
                "top_2_field": top_predictions[1]["field"],
                "top_2_score": top_predictions[1]["score"],
                "top_3_field": top_predictions[2]["field"],
                "top_3_score": top_predictions[2]["score"],
                "inference_ms": round(
                    inference_seconds * 1000,
                    3,
                ),
            }
        )

    relevant_results = [
        result
        for result in model_results
        if result["expected_field"] is not None
    ]

    candidate_correct = sum(
        result["candidate_field"]
        == result["expected_field"]
        for result in relevant_results
    )

    candidate_accuracy = (
        candidate_correct / len(relevant_results)
    )

    macro_f1 = calculate_macro_f1(
        model_results,
        detector.field_names,
    )

    best_threshold_result = select_best_threshold(
        model_results
    )

    mean_inference_ms = (
        sum(inference_times)
        / len(inference_times)
        * 1000
    )

    summary = {
        "model_label": model_label,
        "model_name": model_name,
        "device": detector.device,
        "setup_seconds": round(setup_seconds, 3),
        "mean_inference_ms": round(mean_inference_ms, 3),
        "candidate_correct": candidate_correct,
        "relevant_statements": len(relevant_results),
        "candidate_accuracy": round(candidate_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "selected_threshold": round(
            best_threshold_result["threshold"],
            2,
        ),
        "correct_clinical": (
            best_threshold_result["correct_clinical"]
        ),
        "rejected_clinical": (
            best_threshold_result["rejected_clinical"]
        ),
        "accepted_wrong": (
            best_threshold_result["accepted_wrong"]
        ),
        "rejected_unrelated": (
            best_threshold_result["rejected_unrelated"]
        ),
        "accepted_unrelated": (
            best_threshold_result["accepted_unrelated"]
        ),
        "threshold_accuracy": round(
            best_threshold_result["threshold_accuracy"],
            4,
        ),
    }

    model_summaries.append(summary)

    print(
        "Candidate accuracy: "
        f"{candidate_correct}/{len(relevant_results)} "
        f"({candidate_accuracy:.2%})"
    )
    print(f"Macro F1: {macro_f1:.4f}")
    print(
        "Selected threshold: "
        f"{best_threshold_result['threshold']:.2f}"
    )
    print(
        "Threshold accuracy: "
        f"{best_threshold_result['threshold_accuracy']:.2%}"
    )
    print(
        "Mean inference time: "
        f"{mean_inference_ms:.3f} ms"
    )

    print("Classification errors:")

    errors = [
        result
        for result in relevant_results
        if result["candidate_field"]
        != result["expected_field"]
    ]

    if not errors:
        print("None")
    else:
        for error in errors:
            print(
                f"  Test {error['test_number']:02d}: "
                f"expected={error['expected_field']}, "
                f"candidate={error['candidate_field']}, "
                f"score={error['score']:.4f}"
            )

    # Remove this model before loading the next one.
    del detector
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DETAILS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with DETAILS_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=all_details[0].keys(),
    )

    writer.writeheader()
    writer.writerows(all_details)


with SUMMARY_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=model_summaries[0].keys(),
    )

    writer.writeheader()
    writer.writerows(model_summaries)


print(f"\n{'=' * 70}")
print("FINAL MODEL COMPARISON")
print(f"{'=' * 70}")

print(
    f"{'Model':<12}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Threshold':<12}"
    f"{'End-to-end':<14}"
    f"{'Mean ms'}"
)

print("-" * 70)

for summary in model_summaries:
    print(
        f"{summary['model_label']:<12}"
        f"{summary['candidate_accuracy']:<12.2%}"
        f"{summary['macro_f1']:<12.4f}"
        f"{summary['selected_threshold']:<12.2f}"
        f"{summary['threshold_accuracy']:<14.2%}"
        f"{summary['mean_inference_ms']:.3f}"
    )


print(f"\nDetailed results: {DETAILS_PATH}")
print(f"Comparison summary: {SUMMARY_PATH}")