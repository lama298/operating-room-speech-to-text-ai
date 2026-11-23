from pathlib import Path

from src.audio_processing.live_vad_recorder_v1 import (
    LiveVADRecorderV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "live_vad_test"
)


def print_detected_segment(
    segment_path: Path,
) -> None:
    """
    Print each segment immediately after it is saved.
    """

    print(
        "Callback received:",
        segment_path.name,
    )


def main() -> None:
    """
    Listen for 20 seconds and detect separate spoken
    statements automatically.
    """

    recorder = LiveVADRecorderV1(
        output_directory=OUTPUT_DIRECTORY,
        input_device=None,
        silence_duration_ms=1000,
        speech_threshold=0.5,
    )

    segments = recorder.listen(
        on_segment=print_detected_segment,
        maximum_listening_seconds=20,
    )

    print("\nDetected segment count:", len(segments))

    for segment in segments:
        print(segment)

    if not segments:
        raise AssertionError(
            "No live speech segments were detected."
        )

    print("Live VAD recorder test passed.")


if __name__ == "__main__":
    main()