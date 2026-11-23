import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg


class AudioPreprocessorV1:
    """
    Converts input audio into mono, 16-kHz, PCM WAV.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
    ) -> None:
        """
        Locate FFmpeg and store the target sample rate.
        """

        self.target_sample_rate = (
            target_sample_rate
        )

        self.ffmpeg_path = Path(
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        if not self.ffmpeg_path.exists():
            raise FileNotFoundError(
                f"FFmpeg executable not found: "
                f"{self.ffmpeg_path}"
            )

    def preprocess(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:
        """
        Convert one audio file into mono, 16-kHz,
        uncompressed PCM WAV.
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input audio not found: "
                f"{input_path}"
            )

        if output_path.suffix.lower() != ".wav":
            raise ValueError(
                "The processed output path must end "
                "with .wav."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            str(self.ffmpeg_path),
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(self.target_sample_rate),
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]

        completed_process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        if not output_path.exists():
            raise RuntimeError(
                "FFmpeg finished without creating "
                "the processed WAV file."
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                "The processed WAV file is empty."
            )

        return {
            "input_path": str(
                input_path.resolve()
            ),
            "output_path": str(
                output_path.resolve()
            ),
            "sample_rate": (
                self.target_sample_rate
            ),
            "channels": 1,
            "codec": "pcm_s16le",
            "ffmpeg_return_code": (
                completed_process.returncode
            ),
        }