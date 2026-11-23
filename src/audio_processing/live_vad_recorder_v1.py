from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from time import monotonic
from typing import Callable

import numpy as np
import sounddevice as sd
import torch
from scipy.io import wavfile
from silero_vad import load_silero_vad


class LiveVADRecorderV1:
    """
    Continuously capture microphone audio and save a WAV
    segment whenever Silero detects completed speech.

    This class only handles recording and segmentation.
    It does not run Whisper or clinical extraction.
    """

    def __init__(
        self,
        output_directory: Path | str,
        input_device: int | None = None,
        sample_rate: int = 16000,
        block_size: int = 512,
        speech_threshold: float = 0.5,
        silence_duration_ms: int = 1000,
        pre_speech_ms: int = 300,
        minimum_speech_ms: int = 300,
        maximum_segment_seconds: float = 30.0,
    ):
        """
        Configure the microphone and segmentation rules.
        """

        if sample_rate != 16000:
            raise ValueError(
                "Silero VAD requires 16 kHz audio in "
                "this recorder."
            )

        if block_size != 512:
            raise ValueError(
                "Silero VAD expects 512-sample blocks "
                "for 16 kHz streaming audio."
            )

        if not 0.0 < speech_threshold < 1.0:
            raise ValueError(
                "speech_threshold must be between "
                "0 and 1."
            )

        self.output_directory = Path(
            output_directory
        )

        self.input_device = input_device
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.speech_threshold = (
            speech_threshold
        )

        self.silence_duration_ms = (
            silence_duration_ms
        )

        self.pre_speech_ms = pre_speech_ms

        self.minimum_speech_ms = (
            minimum_speech_ms
        )

        self.maximum_segment_seconds = (
            maximum_segment_seconds
        )

        # One 512-sample block at 16 kHz lasts 32 ms.
        self.block_duration_ms = (
            self.block_size
            / self.sample_rate
            * 1000
        )

        self.silence_block_limit = max(
            1,
            round(
                self.silence_duration_ms
                / self.block_duration_ms
            ),
        )

        self.pre_speech_block_limit = max(
            1,
            round(
                self.pre_speech_ms
                / self.block_duration_ms
            ),
        )

        self.minimum_speech_blocks = max(
            1,
            round(
                self.minimum_speech_ms
                / self.block_duration_ms
            ),
        )

        self.maximum_segment_blocks = max(
            1,
            round(
                self.maximum_segment_seconds
                * self.sample_rate
                / self.block_size
            ),
        )

        # The sounddevice callback places audio here.
        # The main loop removes and analyzes it.
        self.audio_queue: Queue[
            np.ndarray
        ] = Queue()

        self.stop_event = Event()

        self.model = load_silero_vad()

        self.segment_number = 0

    def _audio_callback(
        self,
        input_data: np.ndarray,
        frame_count: int,
        time_information,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Receive one microphone block from PortAudio.

        This callback must remain fast. It only copies
        audio into a queue for later processing.
        """

        if status:
            print(
                "Microphone status:",
                status,
            )

        # Copying is necessary because PortAudio may reuse
        # input_data after this callback returns.
        audio_block = (
            input_data[:, 0]
            .astype(np.float32)
            .copy()
        )

        self.audio_queue.put(
            audio_block
        )

    def _speech_probability(
        self,
        audio_block: np.ndarray,
    ) -> float:
        """
        Return Silero's probability that one block
        contains human speech.
        """

        audio_tensor = torch.from_numpy(
            audio_block
        )

        with torch.no_grad():
            probability = self.model(
                audio_tensor,
                self.sample_rate,
            ).item()

        return float(probability)

    def _save_segment(
        self,
        audio_blocks: list[np.ndarray],
    ) -> Path:
        """
        Join captured blocks and save one speech segment
        as a signed 16-bit PCM WAV file.
        """

        self.segment_number += 1

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        segment_audio = np.concatenate(
            audio_blocks
        )

        pcm_audio = (
            np.clip(
                segment_audio,
                -1.0,
                1.0,
            )
            * 32767
        ).astype(np.int16)

        output_path = (
            self.output_directory
            / (
                f"segment_"
                f"{self.segment_number:04d}.wav"
            )
        )

        wavfile.write(
            output_path,
            self.sample_rate,
            pcm_audio,
        )

        return output_path

    def request_stop(self) -> None:
        """
        Ask the listening loop to stop safely.
        """

        self.stop_event.set()

    def listen(
        self,
        on_segment: Callable[
            [Path],
            None,
        ]
        | None = None,
        maximum_listening_seconds: (
            float | None
        ) = None,
    ) -> list[Path]:
        """
        Listen continuously and save automatically
        completed speech segments.

        on_segment is called after each WAV file is saved.
        """

        self.stop_event.clear()
        self.model.reset_states()

        saved_segments: list[Path] = []

        pre_speech_buffer: deque[
            np.ndarray
        ] = deque(
            maxlen=self.pre_speech_block_limit
        )

        current_segment: list[
            np.ndarray
        ] = []

        recording_speech = False
        silence_block_count = 0
        detected_speech_blocks = 0

        listening_started_at = monotonic()

        print("Listening continuously...")
        print(
            "Speak naturally. Pause for approximately "
            f"{self.silence_duration_ms / 1000:.1f} "
            "seconds to finish a segment."
        )

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                device=self.input_device,
                callback=self._audio_callback,
            ):
                while not self.stop_event.is_set():
                    if (
                        maximum_listening_seconds
                        is not None
                        and (
                            monotonic()
                            - listening_started_at
                        )
                        >= maximum_listening_seconds
                    ):
                        break

                    try:
                        audio_block = (
                            self.audio_queue.get(
                                timeout=0.25
                            )
                        )

                    except Empty:
                        continue

                    probability = (
                        self._speech_probability(
                            audio_block
                        )
                    )

                    contains_speech = (
                        probability
                        >= self.speech_threshold
                    )

                    if not recording_speech:
                        pre_speech_buffer.append(
                            audio_block
                        )

                        if contains_speech:
                            recording_speech = True

                            current_segment = list(
                                pre_speech_buffer
                            )

                            pre_speech_buffer.clear()

                            detected_speech_blocks = 1
                            silence_block_count = 0

                            print(
                                "Speech started "
                                f"(probability "
                                f"{probability:.3f})"
                            )

                        continue

                    current_segment.append(
                        audio_block
                    )

                    if contains_speech:
                        detected_speech_blocks += 1
                        silence_block_count = 0

                    else:
                        silence_block_count += 1

                    silence_finished_segment = (
                        silence_block_count
                        >= self.silence_block_limit
                    )

                    maximum_length_reached = (
                        len(current_segment)
                        >= self.maximum_segment_blocks
                    )

                    if not (
                        silence_finished_segment
                        or maximum_length_reached
                    ):
                        continue

                    if (
                        detected_speech_blocks
                        >= self.minimum_speech_blocks
                    ):
                        segment_path = (
                            self._save_segment(
                                current_segment
                            )
                        )

                        saved_segments.append(
                            segment_path
                        )

                        duration_seconds = (
                            len(current_segment)
                            * self.block_size
                            / self.sample_rate
                        )

                        print(
                            "Speech segment saved:",
                            segment_path,
                        )

                        print(
                            "Segment duration:",
                            round(
                                duration_seconds,
                                3,
                            ),
                            "seconds",
                        )

                        if on_segment is not None:
                            on_segment(
                                segment_path
                            )

                    else:
                        print(
                            "Short sound ignored."
                        )

                    current_segment = []
                    recording_speech = False
                    silence_block_count = 0
                    detected_speech_blocks = 0
                    self.model.reset_states()

        except KeyboardInterrupt:
            print(
                "\nStop requested from keyboard."
            )

        finally:
            # If listening stops while speech is active,
            # preserve it when it contains enough speech.
            if (
                recording_speech
                and detected_speech_blocks
                >= self.minimum_speech_blocks
            ):
                segment_path = self._save_segment(
                    current_segment
                )

                saved_segments.append(
                    segment_path
                )

                print(
                    "Final speech segment saved:",
                    segment_path,
                )

                if on_segment is not None:
                    on_segment(
                        segment_path
                    )

            self.model.reset_states()

        print("Listening stopped.")

        return saved_segments