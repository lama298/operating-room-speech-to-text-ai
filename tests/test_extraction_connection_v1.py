from config.module7.clinical_extraction_pipeline_v1 import (
    ClinicalExtractionPipelineV1,
)


def get_accepted_values(
    pipeline,
    transcript,
):
    """
    Process one transcript and return only accepted
    field-value pairs.
    """

    results = pipeline.process_transcript(
        transcript=transcript,
        top_k=10,
    )

    return {
        result["field"]: result["value"]
        for result in results
        if result.get("status") == "accepted"
    }


def main():
    """
    Test the integrated connection between clause
    segmentation, BioLORD detection, value parsing,
    fallback, and explicit-label recovery.
    """

    print(
        "Creating clinical extraction pipeline..."
    )

    pipeline = ClinicalExtractionPipelineV1()

    test_cases = [
        {
            "name": "basic_preoperative_diagnosis",
            "transcript": (
                "The preoperative diagnosis is "
                "acute appendicitis."
            ),
            "expected": {
                "preoperative_diagnosis": (
                    "acute appendicitis"
                ),
            },
        },
        {
            "name": "case_with_operating_theatre",
            "transcript": (
                "Surgical case Q912 is booked for "
                "November 14, 2027 in operating "
                "theatre 4B."
            ),
            "expected": {
                "case_id": "Q912",
                "date": "2027-11-14",
                "or_room_number": "4B",
            },
        },
        {
            "name": "case_with_surgical_suite",
            "transcript": (
                "Operative case Z305 will occur on "
                "January 8, 2028 in surgical suite 2."
            ),
            "expected": {
                "case_id": "Z305",
                "date": "2028-01-08",
                "or_room_number": "2",
            },
        },
        {
            "name": "nonclinical_meeting",
            "transcript": (
                "The meeting is scheduled for "
                "November 14, 2027."
            ),
            "expected": {},
        },
        {
            "name": "room_cleaning_instruction",
            "transcript": (
                "Please clean operating theatre 4."
            ),
            "expected": {},
        },
        {
            "name": "case_without_identifier",
            "transcript": (
                "The team discussed another "
                "surgical case."
            ),
            "expected": {},
        },
        {
            "name": "patient_entry_not_room_number",
            "transcript": (
                "The patient entered the operating "
                "room at 07:15."
            ),
            "expected": {
                "patient_in_time": "07:15",
            },
        },
                {
            "name": "verified_multiple_allergies",
            "transcript": (
                "Allergy verification was completed. "
                "The patient has known allergies to "
                "iodine and adhesive."
            ),
            "expected": {
                "allergy_status_confirmed": True,
                "known_allergies": [
                    "iodine",
                    "adhesive",
                ],
            },
        },
                {
            "name": "multiple_patient_safety_flags",
            "transcript": (
                "Droplet isolation is indicated and "
                "the patient has an elevated fall risk."
            ),
            "expected": {
                "fall_risk": True,
                "isolation_required": True,
            },
        },
                {
            "name": "primary_and_relief_nurses",
            "transcript": (
                "The scrub nurse was Rima Haddad "
                "and was relieved by Nour Saleh. "
                "The circulating nurse was Karim "
                "Nassar and was relieved by Maya Faris."
            ),
            "expected": {
                "scrub_nurse": "Rima Haddad",
                "relief_scrub_nurse": "Nour Saleh",
                "circulating_nurse": "Karim Nassar",
                "relief_circulating_nurse": "Maya Faris",
            },
        },
                {
            "name": "diagnosis_before_surgery",
            "transcript": (
                "The diagnosis before surgery was "
                "a displaced fracture of the right ulna."
            ),
            "expected": {
                "preoperative_diagnosis": (
                    "displaced fracture of the right ulna"
                ),
            },
        },
                {
            "name": "discussion_before_surgery",
            "transcript": (
                "The team discussed documentation "
                "before surgery."
            ),
            "expected": {},
        },
                {
            "name": "planned_operation_details",
            "transcript": (
                "The planned operation is arthroscopic "
                "reconstruction of the left shoulder."
            ),
            "expected": {
                "procedure_name": (
                    "arthroscopic reconstruction"
                ),
                "procedure_site": "shoulder",
                "laterality": "left",
            },
        },
                {
            "name": "procedure_narrative_and_findings",
            "transcript": (
                "The operation was completed by "
                "removing the lesion and closing the "
                "incision. Operative examination "
                "demonstrated healthy margins and no "
                "vascular injury."
            ),
            "expected": {
                "procedure_description": (
                    "The operation was completed by "
                    "removing the lesion and closing "
                    "the incision"
                ),
                "operative_findings": [
                    "healthy margins",
                    "no vascular injury",
                ],
            },
        },
                {
            "name": "nonstandard_patient_position",
            "transcript": (
                "The patient was positioned in a "
                "reverse jackknife configuration."
            ),
            "expected": {
                "other_position": (
                    "reverse jackknife configuration"
                ),
            },
        },
        {
            "name": "standard_patient_position",
            "transcript": (
                "The patient position was supine."
            ),
            "expected": {
                "patient_position": "supine",
            },
        },
                {
            "name": "irrigation_with_solution",
            "transcript": (
                "The surgical field was irrigated "
                "with normal saline."
            ),
            "expected": {
                "irrigation_used": True,
                "irrigation_solution": [
                    "normal saline",
                ],
            },
        },
        {
            "name": "irrigation_not_required",
            "transcript": (
                "Irrigation was not required."
            ),
            "expected": {
                "irrigation_used": False,
            },
        },
    ]

    passed_tests = 0

    for test_case in test_cases:
        actual = get_accepted_values(
            pipeline=pipeline,
            transcript=test_case["transcript"],
        )

        expected = test_case["expected"]

        if actual != expected:
            raise AssertionError(
                f"{test_case['name']} failed.\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

        passed_tests += 1

        print(
            f"PASSED: {test_case['name']}"
        )

    print(
        "\nFINAL RESULT: "
        f"{passed_tests}/{len(test_cases)} "
        "tests passed"
    )


if __name__ == "__main__":
    main()