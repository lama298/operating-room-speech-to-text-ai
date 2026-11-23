import csv
import json
from pathlib import Path

from semantic_detector import SemanticFieldDetector


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

# Folder containing this evaluation script.
MODULE_DIR = Path(__file__).resolve().parent

# Project root: adjust this only if your folder structure changes.
PROJECT_ROOT = MODULE_DIR.parent.parent

VALIDATION_PATH = MODULE_DIR / "semantic_validation.json"

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "semantic_validation_results_biolord.csv"
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_NAME = "FremyCompany/BioLORD-2023-C"
THRESHOLD = 0.26
TOP_K = 3


# ---------------------------------------------------------
# Load validation data
# ---------------------------------------------------------

with VALIDATION_PATH.open("r", encoding="utf-8") as file:
    validation_data = json.load(file)


# ---------------------------------------------------------
# Load the semantic detector
# ---------------------------------------------------------

detector = SemanticFieldDetector(
    model_name=MODEL_NAME,
    threshold=THRESHOLD,
)


# ---------------------------------------------------------
# Evaluation counters
# ---------------------------------------------------------

results = []

# Clinical/relevant statement counters.
relevant_count = 0
correct_candidate_count = 0
relevant_accepted_count = 0
correct_relevant_final_count = 0

# Unrelated statement counters.
unrelated_count = 0
unrelated_rejected_count = 0


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

for test_number, example in enumerate(validation_data, start=1):
    text = example["text"]
    expected_field = example["expected_field"]

    prediction = detector.predict(
        text,
        top_k=TOP_K,
    )

    candidate_field = prediction["candidate_field"]
    score = prediction["score"]
    accepted = prediction["accepted"]
    predicted_field = prediction["predicted_field"]
    top_predictions = prediction["top_predictions"]

    # Candidate correctness applies only to clinical statements.
    candidate_correct = (
        candidate_field == expected_field
        if expected_field is not None
        else None
    )

    if expected_field is not None:
        # This is a relevant clinical statement.
        relevant_count += 1

        # Did BioLORD rank the correct field first?
        if candidate_correct:
            correct_candidate_count += 1

        # Did the statement pass the threshold?
        if accepted:
            relevant_accepted_count += 1

        # Was the statement accepted under the correct field?
        if predicted_field == expected_field:
            correct_relevant_final_count += 1

    else:
        # This is an unrelated statement.
        unrelated_count += 1

        # Rejection is represented by predicted_field = None.
        if predicted_field is None:
            unrelated_rejected_count += 1

    # Complete end-to-end decision:
    # - Clinical statement: correct field must be accepted.
    # - Unrelated statement: predicted_field must be None.
    final_correct = predicted_field == expected_field

    row = {
        "test_number": test_number,
        "text": text,
        "expected_field": expected_field,
        "candidate_field": candidate_field,
        "candidate_correct": candidate_correct,
        "score": score,
        "accepted_at_current_threshold": accepted,
        "predicted_field": predicted_field,
        "final_correct": final_correct,
        "top_1_field": top_predictions[0]["field"],
        "top_1_score": top_predictions[0]["score"],
        "top_2_field": top_predictions[1]["field"],
        "top_2_score": top_predictions[1]["score"],
        "top_3_field": top_predictions[2]["field"],
        "top_3_score": top_predictions[2]["score"],
    }

    results.append(row)

    expected_display = (
        expected_field
        if expected_field is not None
        else "UNRELATED"
    )

    predicted_display = (
        predicted_field
        if predicted_field is not None
        else "REJECTED"
    )

    print(
        f"{test_number:02d}. "
        f"Expected: {expected_display:<25} "
        f"Candidate: {candidate_field:<25} "
        f"Score: {score:.4f} "
        f"Final: {predicted_display:<25} "
        f"Correct: {final_correct}"
    )


# ---------------------------------------------------------
# Save detailed results to CSV
# ---------------------------------------------------------

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if results:
    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------

candidate_accuracy = (
    correct_candidate_count / relevant_count
    if relevant_count > 0
    else 0.0
)

relevant_acceptance_rate = (
    relevant_accepted_count / relevant_count
    if relevant_count > 0
    else 0.0
)

correct_relevant_final_rate = (
    correct_relevant_final_count / relevant_count
    if relevant_count > 0
    else 0.0
)

unrelated_rejection_rate = (
    unrelated_rejected_count / unrelated_count
    if unrelated_count > 0
    else 0.0
)

final_correct_count = (
    correct_relevant_final_count
    + unrelated_rejected_count
)

final_accuracy = (
    final_correct_count / len(validation_data)
    if validation_data
    else 0.0
)


# ---------------------------------------------------------
# Print evaluation summary
# ---------------------------------------------------------

print("\nEvaluation complete")
print(f"Model: {MODEL_NAME}")
print(f"Threshold: {THRESHOLD}")
print(f"Total statements: {len(validation_data)}")

print("\nCandidate field evaluation")
print(f"Relevant statements: {relevant_count}")
print(
    "Correct relevant candidates: "
    f"{correct_candidate_count}/{relevant_count}"
)
print(f"Candidate accuracy: {candidate_accuracy:.2%}")

print("\nThreshold evaluation")
print(
    "Clinical statements accepted: "
    f"{relevant_accepted_count}/{relevant_count} "
    f"({relevant_acceptance_rate:.2%})"
)
print(
    "Correct clinical decisions after threshold: "
    f"{correct_relevant_final_count}/{relevant_count} "
    f"({correct_relevant_final_rate:.2%})"
)
print(
    "Unrelated statements rejected: "
    f"{unrelated_rejected_count}/{unrelated_count} "
    f"({unrelated_rejection_rate:.2%})"
)

print("\nEnd-to-end evaluation")
print(
    "Final correct decisions: "
    f"{final_correct_count}/{len(validation_data)}"
)
print(
    f"End-to-end threshold accuracy: "
    f"{final_accuracy:.2%}"
)

print(f"\nResults saved to: {RESULTS_PATH}")