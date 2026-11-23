from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from silero_vad import (
    get_speech_timestamps,
    load_silero_vad,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_AUDIO_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "microphone_test"
    / "microphone_capture.wav"
)

SAMPLE_RATE = 16000


def main() -> None:
    """
    Detect speech intervals in the microphone test file.
    """

    if not INPUT_AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Microphone recording not found: "
            f"{INPUT_AUDIO_PATH}"
        )

    print("Loading Silero VAD...")

    model = load_silero_vad()

    print("Reading microphone audio...")

    detected_sample_rate, audio_data = (
    wavfile.read(
        INPUT_AUDIO_PATH
    )
)

    if detected_sample_rate != SAMPLE_RATE:
        raise ValueError(
            "Expected a 16 kHz WAV file, but found "
            f"{detected_sample_rate} Hz."
        )

    # If an audio file has multiple channels, combine
    # them into a single mono waveform.
    if audio_data.ndim > 1:
        audio_data = np.mean(
            audio_data,
            axis=1,
        )

    # Convert signed 16-bit PCM samples from approximately
    # -32768...32767 into floating point -1.0...1.0.
    if np.issubdtype(
        audio_data.dtype,
        np.integer,
    ):
        integer_information = np.iinfo(
            audio_data.dtype
        )

        maximum_amplitude = max(
            abs(integer_information.min),
            integer_information.max,
        )

        normalized_audio = (
            audio_data.astype(np.float32)
            / maximum_amplitude
        )

    else:
        normalized_audio = audio_data.astype(
            np.float32
        )

    waveform = torch.from_numpy(
        normalized_audio
    )

    audio_duration_seconds = (
        len(waveform)
        / SAMPLE_RATE
    )

    print(
        "Audio duration:",
        round(audio_duration_seconds, 3),
        "seconds",
    )

    speech_timestamps = (
        get_speech_timestamps(
            waveform,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            return_seconds=True,
        )
    )

    print("\nDetected speech intervals:")

    for index, interval in enumerate(
        speech_timestamps,
        start=1,
    ):
        start_time = interval["start"]
        end_time = interval["end"]

        duration = end_time - start_time

        print(
            f"Speech {index}: "
            f"{start_time:.3f}s to "
            f"{end_time:.3f}s "
            f"({duration:.3f}s)"
        )

    if not speech_timestamps:
        raise AssertionError(
            "Silero VAD did not detect speech."
        )

    total_speech_seconds = sum(
        interval["end"] - interval["start"]
        for interval in speech_timestamps
    )

    print(
        "\nTotal detected speech:",
        round(total_speech_seconds, 3),
        "seconds",
    )

    print("Silero VAD test passed.")


if __name__ == "__main__":
    main()