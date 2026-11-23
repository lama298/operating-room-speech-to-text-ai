from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "microphone_test"
)

OUTPUT_AUDIO_PATH = (
    OUTPUT_DIRECTORY
    / "microphone_capture.wav"
)

SAMPLE_RATE = 16000
CHANNEL_COUNT = 1
RECORDING_SECONDS = 5

# None uses the operating system's default input device.
# Your current default is the Realtek microphone array.
INPUT_DEVICE = None


def main() -> None:
    """
    Capture five seconds of microphone audio and save
    it as a mono 16 kHz WAV file.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    default_input_device = sd.default.device[0]

    selected_device = (
        default_input_device
        if INPUT_DEVICE is None
        else INPUT_DEVICE
    )

    device_information = sd.query_devices(
        selected_device,
        "input",
    )

    print("Selected input device:")
    print(device_information["name"])

    print(
        f"\nRecording for "
        f"{RECORDING_SECONDS} seconds..."
    )

    print(
        "Speak a clinical sentence now."
    )

    frame_count = (
        SAMPLE_RATE
        * RECORDING_SECONDS
    )

    recording = sd.rec(
        frames=frame_count,
        samplerate=SAMPLE_RATE,
        channels=CHANNEL_COUNT,
        dtype="float32",
        device=INPUT_DEVICE,
    )

    sd.wait()

    print("Recording completed.")

    # Convert the two-dimensional microphone result
    # from shape (samples, 1) into a one-dimensional
    # mono waveform.
    mono_audio = recording.reshape(-1)

    peak_amplitude = float(
        np.max(np.abs(mono_audio))
    )

    root_mean_square = float(
        np.sqrt(
            np.mean(
                np.square(mono_audio)
            )
        )
    )

    print(
        "Peak amplitude:",
        round(peak_amplitude, 6),
    )

    print(
        "RMS amplitude:",
        round(root_mean_square, 6),
    )

    if peak_amplitude < 0.001:
        raise AssertionError(
            "The captured audio is nearly silent. "
            "Check the selected microphone and Windows "
            "microphone permissions."
        )

    # WAV PCM audio is stored as signed 16-bit
    # integers. Clipping prevents overflow.
    pcm_audio = (
        np.clip(
            mono_audio,
            -1.0,
            1.0,
        )
        * 32767
    ).astype(np.int16)

    wavfile.write(
        OUTPUT_AUDIO_PATH,
        SAMPLE_RATE,
        pcm_audio,
    )

    print(
        "\nSaved microphone recording:",
        OUTPUT_AUDIO_PATH,
    )

    print(
        "Sample rate:",
        SAMPLE_RATE,
    )

    print(
        "Samples:",
        len(pcm_audio),
    )


if __name__ == "__main__":
    main()