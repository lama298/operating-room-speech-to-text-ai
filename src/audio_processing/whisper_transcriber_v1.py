import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import whisper


class WhisperTranscriberV1:
    """
    Loads Whisper once and transcribes preprocessed
    mono, 16-kHz WAV audio.
    """

    def __init__(
        self,
        model_name: str = "turbo",
        language: str = "en",
        target_sample_rate: int = 16000,
    ) -> None:
        """
        Select the device and load Whisper once.
        """

        self.model_name = model_name
        self.language = language

        self.target_sample_rate = (
            target_sample_rate
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading Whisper...")
        print("Model:", self.model_name)
        print("Device:", self.device)

        load_start = time.perf_counter()

        self.model = whisper.load_model(
            self.model_name,
            device=self.device,
        )

        self.model_load_time_seconds = (
            time.perf_counter() - load_start
        )

        print(
            "Whisper loaded in:",
            round(
                self.model_load_time_seconds,
                3,
            ),
            "seconds",
        )

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> dict[str, Any]:
        """
        Transcribe one preprocessed WAV file.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Processed audio not found: "
                f"{audio_path}"
            )

        if audio_path.suffix.lower() != ".wav":
            raise ValueError(
                "WhisperTranscriberV1 expects a "
                "preprocessed .wav file."
            )

        audio, sample_rate = sf.read(
            audio_path
        )

        if sample_rate != self.target_sample_rate:
            raise ValueError(
                f"Expected "
                f"{self.target_sample_rate} Hz, "
                f"but received {sample_rate} Hz."
            )

        if audio.ndim > 1:
            audio = audio.mean(
                axis=1
            )

        audio = audio.astype(
            "float32"
        )

        audio_duration_seconds = (
            len(audio) / sample_rate
        )

        if audio_duration_seconds <= 0:
            raise ValueError(
                "The processed audio contains no samples."
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        inference_start = (
            time.perf_counter()
        )

        result = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            verbose=False,
            fp16=(
                self.device == "cuda"
            ),
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        inference_time_seconds = (
            time.perf_counter()
            - inference_start
        )

        transcript = result[
            "text"
        ].strip()

        real_time_factor = (
            inference_time_seconds
            / audio_duration_seconds
        )

        return {
            "transcript": transcript,
            "audio_path": str(
                audio_path.resolve()
            ),
            "audio_duration_seconds": round(
                audio_duration_seconds,
                4,
            ),
            "inference_time_seconds": round(
                inference_time_seconds,
                4,
            ),
            "real_time_factor": round(
                real_time_factor,
                4,
            ),
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "sample_rate": sample_rate,
        }