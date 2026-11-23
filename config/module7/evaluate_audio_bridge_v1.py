import csv
from pathlib import Path

from semantic_detector_v2 import SemanticFieldDetectorV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TURBO_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "whisper_benchmark"
    / "turbo_pilot_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "audio_bridge_results_v1.csv"
)


def load_turbo_results():
    """Load the saved Whisper Turbo benchmark results."""

    with TURBO_RESULTS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        row["filename"]: row
        for row in rows
    }


# Gold-standard clause annotations for the 20 pilot recordings.
#
# Compound recordings produce more than one clause because the
# current semantic detector predicts one field per clause.
BRIDGE_CASES = [
    # Standard recordings
    {
        "filename": "STD_001.m4a",
        "clause_id": "diagnosis",
        "expected_field": "preoperative_diagnosis",
        "reference_clause": (
            "The preoperative diagnosis is acute appendicitis."
        ),
        "whisper_clause": (
            "The preoperative diagnosis is acute appendicitis."
        ),
    },
    {
        "filename": "STD_002.m4a",
        "clause_id": "procedure",
        "expected_field": "procedure_name",
        "reference_clause": (
            "The procedure is laparoscopic appendectomy."
        ),
        "whisper_clause": (
            "The procedure is laparoscopic appendectomy."
        ),
    },
    {
        "filename": "STD_003.m4a",
        "clause_id": "anesthesia",
        "expected_field": "anesthesia_type",
        "reference_clause": (
            "General anesthesia was administered."
        ),
        "whisper_clause": (
            "General Anastasia was administered."
        ),
    },
    {
        "filename": "STD_004.m4a",
        "clause_id": "position",
        "expected_field": "patient_position",
        "reference_clause": (
            "The patient was positioned supine."
        ),
        "whisper_clause": (
            "The patient was positioned supine."
        ),
    },
    {
        "filename": "STD_005.m4a",
        "clause_id": "medication",
        "expected_field": "medication_details",
        "reference_clause": (
            "One gram of cefazolin was administered."
        ),
        "whisper_clause": (
            "One gram of CFAS loan was administered."
        ),
    },
    {
        "filename": "STD_006.m4a",
        "clause_id": "blood_loss",
        "expected_field": "estimated_blood_loss_ml",
        "reference_clause": (
            "Estimated blood loss was fifty milliliters."
        ),
        "whisper_clause": (
            "Estimated blood loss was 50 milliliters."
        ),
    },
    {
        "filename": "STD_007.m4a",
        "clause_id": "counts",
        "expected_field": "counts_correct",
        "reference_clause": (
            "The final instrument, sponge, and needle "
            "counts were correct."
        ),
        "whisper_clause": (
            "The final instrument sponge and needle "
            "counts were correct."
        ),
    },
    {
        "filename": "STD_008.m4a",
        "clause_id": "destination",
        "expected_field": "transfer_destination",
        "reference_clause": (
            "The patient was transferred to the "
            "post-anesthesia care unit."
        ),
        "whisper_clause": (
            "The patient was transferred to the "
            "post-anesthesia care unit."
        ),
    },
    {
        "filename": "STD_008.m4a",
        "clause_id": "status",
        "expected_field": "patient_status",
        "reference_clause": (
            "The patient was in stable condition."
        ),
        "whisper_clause": (
            "The patient was in stable condition."
        ),
    },

    # MIX_001: diagnosis + procedure + anesthesia
    {
        "filename": "MIX_001.m4a",
        "clause_id": "diagnosis",
        "expected_field": "preoperative_diagnosis",
        "reference_clause": (
            "The pre-operative diagnosis is acute appendicitis."
        ),
        "whisper_clause": (
            "The preoperative diagnosis is acute appendicitis."
        ),
    },
    {
        "filename": "MIX_001.m4a",
        "clause_id": "procedure",
        "expected_field": "procedure_name",
        "reference_clause": (
            "The procedure is laparoscopic appendectomy."
        ),
        "whisper_clause": (
            "The procedure is laparoscopic appendectomy."
        ),
    },
    {
        "filename": "MIX_001.m4a",
        "clause_id": "anesthesia",
        "expected_field": "anesthesia_type",
        "reference_clause": "Under general anesthesia.",
        "whisper_clause": "Under general anesthesia.",
    },

    # MIX_002: position + antibiotic
    {
        "filename": "MIX_002.m4a",
        "clause_id": "position",
        "expected_field": "patient_position",
        "reference_clause": (
            "The patient was positioned supine."
        ),
        "whisper_clause": (
            "The patient was positioned supine."
        ),
    },
    {
        "filename": "MIX_002.m4a",
        "clause_id": "antibiotic",
        "expected_field": "antibiotic_prophylaxis_given",
        "reference_clause": (
            "One gram of cefazolin was administered "
            "before incision."
        ),
        "whisper_clause": (
            "1 gram of Cifazolin was administered "
            "before the incision."
        ),
    },

    # MIX_003: specimen + pathology + blood loss
    {
        "filename": "MIX_003.m4a",
        "clause_id": "specimen",
        "expected_field": "specimen_collected",
        "reference_clause": (
            "The appendix specimen was collected."
        ),
        "whisper_clause": (
            "The appendix specimen was collected."
        ),
    },
    {
        "filename": "MIX_003.m4a",
        "clause_id": "pathology",
        "expected_field": "laboratory_requests",
        "reference_clause": (
            "The appendix specimen was sent to pathology."
        ),
        "whisper_clause": (
            "The appendix specimen was sent to pathology."
        ),
    },
    {
        "filename": "MIX_003.m4a",
        "clause_id": "blood_loss",
        "expected_field": "estimated_blood_loss_ml",
        "reference_clause": (
            "Estimated blood loss was fifty milliliters."
        ),
        "whisper_clause": (
            "Estimated blood loss was 50 milliliters."
        ),
    },

    # MIX_004: counts + status
    {
        "filename": "MIX_004.m4a",
        "clause_id": "counts",
        "expected_field": "counts_correct",
        "reference_clause": (
            "The final counts were correct."
        ),
        "whisper_clause": (
            "The final counts were correct."
        ),
    },
    {
        "filename": "MIX_004.m4a",
        "clause_id": "status",
        "expected_field": "patient_status",
        "reference_clause": (
            "The patient remained stable."
        ),
        "whisper_clause": (
            "The patient remained stable."
        ),
    },

    # MIX_005: findings + complications
    {
        "filename": "MIX_005.m4a",
        "clause_id": "findings",
        "expected_field": "operative_findings",
        "reference_clause": (
            "The operative findings showed an inflamed "
            "appendix without perforation."
        ),
        "whisper_clause": (
            "The operative findings showed an inflamed "
            "appendix without perforation."
        ),
    },
    {
        "filename": "MIX_005.m4a",
        "clause_id": "complications",
        "expected_field": "complications",
        "reference_clause": (
            "No intraoperative complications occurred."
        ),
        "whisper_clause": (
            "No intraoperative complications occurred."
        ),
    },

    # Unexpected-event recordings
    {
        "filename": "UNEXP_001.m4a",
        "clause_id": "unexpected_event",
        "expected_field": "unexpected_event",
        "reference_clause": (
            "Unexpected arterial bleeding occurred "
            "and was controlled with clips."
        ),
        "whisper_clause": (
            "Unexpected arterial bleeding occurred "
            "and was controlled with clips."
        ),
    },
    {
        "filename": "UNEXP_002.m4a",
        "clause_id": "unexpected_event",
        "expected_field": "unexpected_event",
        "reference_clause": (
            "The laparoscopic procedure was converted "
            "to an open procedure because of difficult exposure."
        ),
        "whisper_clause": (
            "The laparoscopic procedure was converted "
            "to an open procedure because of difficult exposure."
        ),
    },
    {
        "filename": "UNEXP_003.m4a",
        "clause_id": "unexpected_event",
        "expected_field": "unexpected_event",
        "reference_clause": (
            "A sudden decrease in blood pressure was observed "
            "and the anesthesia team was notified."
        ),
        "whisper_clause": (
            "A sudden decrease in blood pressure was observed "
            "and the anesthesia team was notified."
        ),
    },
    {
        "filename": "UNEXP_004.m4a",
        "clause_id": "discrepancy",
        "expected_field": "count_discrepancy",
        "reference_clause": (
            "A final count discrepancy was identified."
        ),
        "whisper_clause": (
            "A final count discrepancy was identified."
        ),
    },
    {
        "filename": "UNEXP_004.m4a",
        "clause_id": "count_action",
        "expected_field": "count_actions",
        "reference_clause": (
            "The surgical field was checked again."
        ),
        "whisper_clause": (
            "The surgical field was checked again."
        ),
    },

    # Non-clinical recordings
    {
        "filename": "NON_001.m4a",
        "clause_id": "non_clinical",
        "expected_field": "UNRELATED",
        "reference_clause": (
            "Please move the light slightly to the left."
        ),
        "whisper_clause": (
            "Please move the light slightly to the left."
        ),
    },
    {
        "filename": "NON_002.m4a",
        "clause_id": "non_clinical",
        "expected_field": "UNRELATED",
        "reference_clause": (
            "Can someone adjust the monitor?"
        ),
        "whisper_clause": (
            "Can someone adjust the monitor?"
        ),
    },
    {
        "filename": "NON_003.m4a",
        "clause_id": "non_clinical",
        "expected_field": "UNRELATED",
        "reference_clause": (
            "Thank you, that is fine."
        ),
        "whisper_clause": (
            "Thank you, that's fine."
        ),
    },
]


def decision_is_correct(expected_field, result):
    """Evaluate a clinical prediction or unrelated rejection."""

    if expected_field == "UNRELATED":
        return result["predicted_field"] is None

    return result["predicted_field"] == expected_field


def main():
    turbo_rows = load_turbo_results()

    # Use the locked development threshold.
    detector = SemanticFieldDetectorV2(
        threshold=0.28,
    )

    detector_fields = set(detector.field_names)
    results = []

    for case_number, case in enumerate(
        BRIDGE_CASES,
        start=1,
    ):
        filename = case["filename"]
        expected_field = case["expected_field"]

        # Ensure the bridge case belongs to a real benchmark file.
        if filename not in turbo_rows:
            raise ValueError(
                f"Missing Turbo result for {filename}"
            )

        # Ensure expected fields exist in the detector taxonomy.
        if (
            expected_field != "UNRELATED"
            and expected_field not in detector_fields
        ):
            raise ValueError(
                f"Unknown expected field: {expected_field}"
            )

        reference_result = detector.predict(
            case["reference_clause"],
            top_k=3,
        )

        whisper_result = detector.predict(
            case["whisper_clause"],
            top_k=3,
        )

        reference_correct = decision_is_correct(
            expected_field,
            reference_result,
        )

        whisper_correct = decision_is_correct(
            expected_field,
            whisper_result,
        )

        # True means Whisper introduced a semantic failure.
        asr_induced_error = (
            reference_correct
            and not whisper_correct
        )

        row = {
            "case_number": case_number,
            "filename": filename,
            "clause_id": case["clause_id"],
            "source_wer": turbo_rows[filename]["wer"],
            "expected_field": expected_field,
            "reference_clause": case["reference_clause"],
            "reference_candidate": (
                reference_result["candidate_field"]
            ),
            "reference_prediction": (
                reference_result["predicted_field"]
                or "REJECTED"
            ),
            "reference_score": reference_result["score"],
            "reference_correct": reference_correct,
            "whisper_clause": case["whisper_clause"],
            "whisper_candidate": (
                whisper_result["candidate_field"]
            ),
            "whisper_prediction": (
                whisper_result["predicted_field"]
                or "REJECTED"
            ),
            "whisper_score": whisper_result["score"],
            "whisper_correct": whisper_correct,
            "asr_induced_error": asr_induced_error,
        }

        results.append(row)

        print(
            f"{case_number:02d}. "
            f"{filename} | {case['clause_id']}"
        )
        print("    Expected:", expected_field)
        print(
            "    Reference:",
            row["reference_prediction"],
            row["reference_score"],
            reference_correct,
        )
        print(
            "    Whisper:  ",
            row["whisper_prediction"],
            row["whisper_score"],
            whisper_correct,
        )
        print(
            "    ASR-induced error:",
            asr_induced_error,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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

    total = len(results)

    reference_correct_count = sum(
        row["reference_correct"]
        for row in results
    )

    whisper_correct_count = sum(
        row["whisper_correct"]
        for row in results
    )

    asr_induced_count = sum(
        row["asr_induced_error"]
        for row in results
    )

    print("\nAUDIO-BRIDGE EVALUATION")
    print("=" * 60)
    print("Clause cases:", total)

    print(
        "Reference-clause accuracy:",
        f"{reference_correct_count}/{total}",
        f"({reference_correct_count / total:.2%})",
    )

    print(
        "Whisper-clause accuracy:",
        f"{whisper_correct_count}/{total}",
        f"({whisper_correct_count / total:.2%})",
    )

    print(
        "ASR-induced semantic errors:",
        asr_induced_count,
    )

    print(
        "Results saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()