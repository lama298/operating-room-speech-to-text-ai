import json
from pathlib import Path

from clinical_extraction_pipeline_v1 import (
    ClinicalExtractionPipelineV1,
)
from json_mapper import (
    load_schema,
    map_extraction_results,
    save_schema,
)


if __name__ == "__main__":
    pipeline = ClinicalExtractionPipelineV1()

    test_transcripts = [
        (
            "The preoperative diagnosis is acute appendicitis "
            "and the procedure is laparoscopic appendectomy "
            "under general anesthesia."
        ),
        (
            "The patient was positioned supine and one gram "
            "of cefazolin was administered before incision."
        ),
        (
            "The appendix specimen was collected and sent "
            "to pathology with estimated blood loss of "
            "50 milliliters."
        ),
        (
            "The final counts were correct, the procedure "
            "was completed successfully, and the patient "
            "remained stable."
        ),
        (
            "The final instrument, sponge, and needle "
            "counts were correct."
        ),
    ]

    # Collect extraction results from all transcripts.
    all_extraction_results = []

    for transcript in test_transcripts:
        transcript_results = (
            pipeline.process_transcript(
                transcript,
                top_k=3,
            )
        )

        all_extraction_results.extend(
            transcript_results
        )

    # Load a clean Schema V2 template.
    schema = load_schema()

    # Insert only accepted, validated values.
    mapping_result = map_extraction_results(
        schema=schema,
        extraction_results=all_extraction_results,
    )

    populated_schema = mapping_result["schema"]

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    output_path = (
        project_root
        / "outputs"
        / "structured_case_v1.json"
    )

    save_schema(
        schema=populated_schema,
        output_path=output_path,
    )

    print("\n" + "=" * 70)
    print("MAPPING SUMMARY")
    print(
        "Total extraction results:",
        mapping_result["total_results"],
    )
    print(
        "Successfully mapped:",
        mapping_result["mapped_count"],
    )

    print("\nMAPPING LOG")

    for index, log_entry in enumerate(
        mapping_result["mapping_log"],
        start=1,
    ):
        print(f"\n{index}. Field:", log_entry["field"])
        print("   Status:", log_entry["status"])
        print("   Reason:", log_entry["reason"])

        if log_entry["status"] == "mapped":
            print(
                "   Schema path:",
                log_entry["schema_path"],
            )
            print(
                "   Value:",
                log_entry["value"],
            )

    print("\n" + "=" * 70)
    print("POPULATED RELEVANT SECTIONS")

    relevant_sections = {
        "diagnosis_and_procedure": (
            populated_schema[
                "diagnosis_and_procedure"
            ]
        ),
        "anesthesia": populated_schema[
            "anesthesia"
        ],
        "patient_position": populated_schema[
            "patient_position"
        ],
        "specimens_and_laboratory": (
            populated_schema[
                "specimens_and_laboratory"
            ]
        ),
        "medications": populated_schema[
            "medications"
        ],
        "fluids_and_blood": populated_schema[
            "fluids_and_blood"
        ],
        "counts": populated_schema[
            "counts"
        ],
        "patient_outcome": populated_schema[
            "patient_outcome"
        ],
    }

    print(
        json.dumps(
            relevant_sections,
            indent=4,
            ensure_ascii=False,
        )
    )

    print("\nSaved record:")
    print(output_path)