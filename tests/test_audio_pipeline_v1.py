from pathlib import Path

from src.audio_processing.audio_preprocessor_v1 import (
    AudioPreprocessorV1,
)

from src.audio_processing.whisper_transcriber_v1 import (
    WhisperTranscriberV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_AUDIO_PATH = (
    PROJECT_ROOT
    / "audio"
    / "pilot"
    / "raw"
    / "STD_001.m4a"
)

PROCESSED_AUDIO_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "audio_pipeline_test"
    / "STD_001_processed.wav"
)


if __name__ == "__main__":
    preprocessor = AudioPreprocessorV1()

    preprocessing_result = (
        preprocessor.preprocess(
            input_path=INPUT_AUDIO_PATH,
            output_path=PROCESSED_AUDIO_PATH,
        )
    )

    print("\nPREPROCESSING RESULT")
    print("=" * 70)

    for key, value in (
        preprocessing_result.items()
    ):
        print(f"{key}: {value}")

    transcriber = WhisperTranscriberV1(
        model_name="turbo",
        language="en",
    )

    transcription_result = (
        transcriber.transcribe(
            PROCESSED_AUDIO_PATH
        )
    )

    print("\nTRANSCRIPTION RESULT")
    print("=" * 70)

    for key, value in (
        transcription_result.items()
    ):
        print(f"{key}: {value}")