import csv
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "semantic_validation_results.csv"
)

THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.36,
    0.38,
    0.40,
    0.42,
    0.45,
    0.50,
    0.55,
    0.60,
]


# Load the results generated in Step 6C.
with RESULTS_PATH.open(
    "r",
    encoding="utf-8-sig",
) as file:
    validation_results = list(csv.DictReader(file))


print(
    f"{'Threshold':<11}"
    f"{'Correct clinical':<19}"
    f"{'Rejected clinical':<19}"
    f"{'Accepted wrong':<17}"
    f"{'Rejected unrelated':<20}"
    f"{'Accepted unrelated':<20}"
    f"{'Accuracy'}"
)

print("-" * 115)


for threshold in THRESHOLDS: #process one threshold at a time
    correct_clinical = 0 #relevant statement accepted with the correct field
    rejected_clinical = 0 #relevant statement rejected because score is too low
    accepted_wrong = 0 #relevant statement accepted with wrong field
    rejected_unrelated = 0 # unrelated speech correctly rejected
    accepted_unrelated = 0 #unrealted speech incorrectly accrepted
    #=0 means it resets the counter each time
#accuracy= correct clinical+rejected clinical/48
    for result in validation_results: #examine all 48 results
        #extract all important values
        expected_field = result["expected_field"].strip()
        candidate_field = result["candidate_field"].strip()
        score = float(result["score"])

        accepted = score >= threshold #decide whether statement is accepted
        is_relevant = expected_field != "" #determine whether it is clinical or unrelated

        if is_relevant: #categorize clinical statements
            candidate_correct = candidate_field == expected_field

            if accepted and candidate_correct:
                correct_clinical += 1
            elif accepted and not candidate_correct:
                accepted_wrong += 1
            else:
                rejected_clinical += 1

        else:
            if accepted:
                accepted_unrelated += 1
            else:
                rejected_unrelated += 1

    total_correct = correct_clinical + rejected_unrelated# (reject unrelated statemeents)
    total_examples = len(validation_results)
    accuracy = total_correct / total_examples

    print(
        f"{threshold:<11.2f}"
        f"{correct_clinical:<19}"
        f"{rejected_clinical:<19}"
        f"{accepted_wrong:<17}"
        f"{rejected_unrelated:<20}"
        f"{accepted_unrelated:<20}"
        f"{accuracy:.2%}"
    )
    """Load the 48 previously calculated results
        ↓
Select threshold 0.25
        ↓
Check all 48 statements
        ↓
Count correct/rejected/incorrect outcomes
        ↓
Calculate accuracy
        ↓
Print one table row
        ↓
Repeat with threshold 0.30, 0.35, ..., 0.60
    """