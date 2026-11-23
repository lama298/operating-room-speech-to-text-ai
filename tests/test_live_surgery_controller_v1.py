from pathlib import Path

from src.live_surgery_controller_v1 import (
    LiveSurgeryControllerV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "live_surgery_demo"
)

RECORD_ID = "LIVE_DEMO_TEST_001"


def main() -> None:
    """
    Test continuous recording, automatic segmentation,
    extraction, accumulation, MongoDB, JSON, and HTML.
    """

    print("Initializing live surgical case...")

    controller = LiveSurgeryControllerV1(
        record_id=RECORD_ID,
        output_directory=OUTPUT_DIRECTORY,
        input_device=None,
        silence_duration_ms=1000,
        speech_threshold=0.5,
        top_k=3,
    )

    try:
        print(
            "\nWait for the message "
            "'Listening continuously' before speaking."
        )

        results = controller.run_live_capture(
            maximum_listening_seconds=30,
        )

        print(
            "\nProcessed segment count:",
            len(results),
        )

        current_record = (
            controller.get_current_record()
        )

        metadata = current_record[
            "processing_metadata"
        ]

        print(
            "Utterance count:",
            metadata["utterance_count"],
        )

        print(
            "Accepted field count:",
            metadata["accepted_field_count"],
        )

        print(
            "Combined transcript:",
            metadata["combined_transcript"],
        )

        if metadata["accepted_field_count"] == 0:
            raise AssertionError(
                "No clinical fields were accepted "
                "from the live microphone."
            )

        print("\nFinalizing live surgical case...")

        final_result = (
            controller.finalize_case()
        )

        print(
            "Structured JSON:",
            final_result[
                "saved_schema_path"
            ],
        )

        print(
            "Complete HTML report:",
            final_result[
                "report_output_path"
            ],
        )

        print(
            "Live surgery controller "
            "test passed."
        )

    finally:
        controller.close()

        print("MongoDB client closed.")


if __name__ == "__main__":
    main()