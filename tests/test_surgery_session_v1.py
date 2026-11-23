import json
from pathlib import Path

from src.surgery_session_v1 import (
    SurgerySessionV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIRECTORY = (
    PROJECT_ROOT
    / "audio"
    / "final_test"
    / "raw"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "surgery_session_test"
)

FIRST_AUDIO_PATH = (
    AUDIO_DIRECTORY
    / "STD_013.m4a"
)

SECOND_AUDIO_PATH = (
    AUDIO_DIRECTORY
    / "STD_019.m4a"
)

FIRST_PROCESSED_PATH = (
    OUTPUT_DIRECTORY
    / "segment_001_processed.wav"
)

SECOND_PROCESSED_PATH = (
    OUTPUT_DIRECTORY
    / "segment_002_processed.wav"
)

FINAL_SCHEMA_PATH = (
    OUTPUT_DIRECTORY
    / "DEMO_SESSION_TEST_001.json"
)
FINAL_REPORT_PATH=(
    OUTPUT_DIRECTORY
    /"clinical_report_DEMO_SESSION_TEST_001.html"
)

RECORD_ID = "DEMO_SESSION_TEST_001"


def main() -> None:
    """
    Verify that multiple audio segments populate one
    accumulated surgical record.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Creating surgery session...")

    session = SurgerySessionV1(
        record_id=RECORD_ID,
        whisper_model_name="turbo",
        language="en",
    )

    try:
        print("\nProcessing segment 1...")

        first_result = (
            session.process_audio_segment(
                input_audio_path=(
                    FIRST_AUDIO_PATH
                ),
                processed_audio_path=(
                    FIRST_PROCESSED_PATH
                ),
                top_k=3,
            )
        )

        print(
            "Transcript 1:",
            first_result["transcript"],
        )

        print(
            "Accepted fields 1:",
            [
                result["field"]
                for result in first_result[
                    "accepted_extractions"
                ]
            ],
        )

        record_after_first = (
            session.get_current_record()
        )

        first_blood_loss = (
            record_after_first[
                "fluids_and_blood"
            ][
                "estimated_blood_loss_ml"
            ]
        )

        print(
            "Blood loss after segment 1:",
            first_blood_loss,
        )

        assert first_blood_loss == 125.0

        print("\nProcessing segment 2...")

        second_result = (
            session.process_audio_segment(
                input_audio_path=(
                    SECOND_AUDIO_PATH
                ),
                processed_audio_path=(
                    SECOND_PROCESSED_PATH
                ),
                top_k=3,
            )
        )

        print(
            "Transcript 2:",
            second_result["transcript"],
        )

        print(
            "Accepted fields 2:",
            [
                result["field"]
                for result in second_result[
                    "accepted_extractions"
                ]
            ],
        )

        accumulated_record = (
            session.get_current_record()
        )

        accumulated_blood_loss = (
            accumulated_record[
                "fluids_and_blood"
            ][
                "estimated_blood_loss_ml"
            ]
        )

        accumulated_destination = (
            accumulated_record[
                "patient_outcome"
            ][
                "destination"
            ]
        )

        print("\nACCUMULATED VALUES")
        print("=" * 70)

        print(
            "Estimated blood loss:",
            accumulated_blood_loss,
        )

        print(
            "Transfer destination:",
            accumulated_destination,
        )

        # The value from segment 1 must still exist.
        assert accumulated_blood_loss == 125.0

        # The value from segment 2 must also exist.
        assert accumulated_destination == "icu"

        metadata = accumulated_record[
            "processing_metadata"
        ]

        print(
            "Utterance count:",
            metadata["utterance_count"],
        )

        print(
            "Combined transcript:",
            metadata["combined_transcript"],
        )

        assert metadata["utterance_count"] == 2

        assert len(
            metadata["utterances"]
        ) == 2

        print("\nFinalizing surgical case...")

        finalization_result = (
            session.finalize_case(
                schema_output_path=(
                    FINAL_SCHEMA_PATH
                ),
                report_output_path=(
                    FINAL_REPORT_PATH
                ),
            )
        )

        print(
            json.dumps(
                finalization_result,
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )
        if not FINAL_REPORT_PATH.exists():
            raise AssertionError(
                "the complete surgical HTML report"
                "was not created"
            )
        html_report= FINAL_REPORT_PATH.read_text(
            encoding="utf-8"
        ).lower()
        assert "demo_session_test_001" in html_report
        assert "125" in html_report
        assert "icu" in html_report
        print(
            "complete surgical HTML report: ",
            FINAL_REPORT_PATH
        )

        stored_record = (
            session.get_stored_record()
        )

        if stored_record is None:
            raise AssertionError(
                "The accumulated MongoDB record "
                "was not retrieved."
            )

        stored_blood_loss = (
            stored_record[
                "fluids_and_blood"
            ][
                "estimated_blood_loss_ml"
            ]
        )

        stored_destination = (
            stored_record[
                "patient_outcome"
            ][
                "destination"
            ]
        )

        stored_metadata = stored_record[
            "processing_metadata"
        ]
        assert(stored_metadata["report_generated"] is True)

        assert stored_blood_loss == 125.0
        assert stored_destination == "icu"

        assert (
            stored_metadata["session_status"]
            == "finalized"
        )

        assert (
            stored_metadata["utterance_count"]
            == 2
        )

        print(
            "\nSurgery-session accumulation "
            "test passed."
        )

    finally:
        session.close()

        print(
            "MongoDB client closed."
        )


if __name__ == "__main__":
    main()