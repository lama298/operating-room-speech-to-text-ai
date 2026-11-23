import re
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from src.audio_processing.live_vad_recorder_v1 import (
    LiveVADRecorderV1,
)
from src.surgery_session_v1 import (
    SurgerySessionV1,
)


class LiveSurgeryControllerV1:
    """
    Connect continuous microphone segmentation with one
    accumulated surgical documentation session.
    """

    def __init__(
        self,
        record_id: str,
        output_directory: Path | str,
        input_device: int | None = None,
        whisper_model_name: str = "turbo",
        language: str = "en",
        top_k: int = 3,
        silence_duration_ms: int = 1000,
        speech_threshold: float = 0.5,
    ):
        """
        Initialize one complete live surgical case.
        """

        self.record_id = record_id.strip()

        if not self.record_id:
            raise ValueError(
                "record_id cannot be empty."
            )

        self.output_directory = Path(
            output_directory
        )

        # Use a filesystem-safe version of the ID for
        # directory and filename creation.
        self.safe_record_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            self.record_id,
        )

        self.case_directory = (
            self.output_directory
            / self.safe_record_id
        )

        self.raw_segment_directory = (
            self.case_directory
            / "raw_segments"
        )

        self.processed_segment_directory = (
            self.case_directory
            / "processed_segments"
        )

        self.final_output_directory = (
            self.case_directory
            / "final"
        )

        self.raw_segment_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed_segment_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.final_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.top_k = top_k

        # This object owns the accumulated Schema V2
        # record and the MongoDB connection.
        self.session = SurgerySessionV1(
            record_id=self.record_id,
            whisper_model_name=(
                whisper_model_name
            ),
            language=language,
        )

        # This object owns microphone capture and
        # automatic speech-boundary detection.
        self.recorder = LiveVADRecorderV1(
            output_directory=(
                self.raw_segment_directory
            ),
            input_device=input_device,
            silence_duration_ms=(
                silence_duration_ms
            ),
            speech_threshold=(
                speech_threshold
            ),
        )

        # The recorder places completed WAV paths here.
        # The worker removes and processes them.
        self.segment_queue: Queue[
            Path | None
        ] = Queue()

        self.processing_results: list[
            dict[str, Any]
        ] = []

        self.processing_errors: list[
            str
        ] = []

        self.processing_worker: (
            Thread | None
        ) = None

    def _enqueue_segment(
        self,
        segment_path: Path,
    ) -> None:
        """
        Receive a completed segment from the recorder
        without blocking microphone capture.
        """

        
        self.segment_queue.put(
            segment_path
        )

    def _process_segment_worker(self) -> None:
        """
        Process queued audio segments sequentially.

        Only this worker modifies the accumulated case,
        preventing two segments from changing Schema V2
        simultaneously.
        """

        while True:
            segment_path = (
                self.segment_queue.get()
            )

            try:
                # None is the shutdown signal.
                if segment_path is None:
                    return

                processed_path = (
                    self.processed_segment_directory
                    / (
                        f"{segment_path.stem}"
                        "_processed.wav"
                    )
                )

                
                result = (
                    self.session
                    .process_audio_segment(
                        input_audio_path=(
                            segment_path
                        ),
                        processed_audio_path=(
                            processed_path
                        ),
                        top_k=self.top_k,
                        persist=True,
                    )
                )

                self.processing_results.append(
                    result
                )

                
            except Exception as error:
                error_message = (
                    f"{segment_path.name}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                self.processing_errors.append(
                    error_message
                )

            finally:
                self.segment_queue.task_done()

    def _start_processing_worker(
        self,
    ) -> None:
        """
        Start the background segment-processing worker.
        """

        if (
            self.processing_worker is not None
            and self.processing_worker.is_alive()
        ):
            raise RuntimeError(
                "The processing worker is already "
                "running."
            )

        self.processing_worker = Thread(
            target=self._process_segment_worker,
            name="clinical-segment-worker",
            daemon=False,
        )

        self.processing_worker.start()

    def _stop_processing_worker(
        self,
    ) -> None:
        """
        Wait for all queued segments and then stop
        the processing worker.
        """

        # Place the shutdown marker after every segment
        # already present in the FIFO queue.
        self.segment_queue.put(
            None
        )

        self.segment_queue.join()

        if self.processing_worker is not None:
            self.processing_worker.join()

    def run_live_capture(
        self,
        maximum_listening_seconds: (
            float | None
        ) = None,
    ) -> list[dict[str, Any]]:
        """
        Start automatic microphone capture and process
        every completed speech segment.
        """

        self._start_processing_worker()

        try:
            self.recorder.listen(
                on_segment=(
                    self._enqueue_segment
                ),
                maximum_listening_seconds=(
                    maximum_listening_seconds
                ),
            )

        finally:
            # Even if microphone capture fails, finish
            # or report every segment already queued.
            self._stop_processing_worker()

        if self.processing_errors:
            formatted_errors = "\n".join(
                self.processing_errors
            )

            raise RuntimeError(
                "One or more live segments failed:\n"
                f"{formatted_errors}"
            )

        return self.processing_results
    def process_uploaded_recording(
    self,
    audio_path: Path | str,
) -> dict[str, Any]:
        """
        Process one complete saved recording through the
        existing transcription, extraction, validation,
        MongoDB, and report workflow.
        """

        input_audio_path = Path(audio_path)

        if not input_audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist: {input_audio_path}"
            )

        processed_audio_path = (
            self.processed_segment_directory
            / f"{input_audio_path.stem}_processed.wav"
        )

        result = self.session.process_audio_segment(
            input_audio_path=input_audio_path,
            processed_audio_path=processed_audio_path,
            top_k=self.top_k,
            persist=True,
        )

        self.processing_results.append(result)

        return result

    def finalize_case(
        self,
    ) -> dict[str, Any]:
        """
        Finalize JSON, MongoDB, and the complete HTML
        report for this surgical case.
        """

        schema_output_path = (
            self.final_output_directory
            / (
                f"{self.safe_record_id}"
                "_structured.json"
            )
        )

        report_output_path = (
            self.final_output_directory
            / (
                "clinical_report_"
                f"{self.safe_record_id}.html"
            )
        )

        return self.session.finalize_case(
            schema_output_path=(
                schema_output_path
            ),
            report_output_path=(
                report_output_path
            ),
        )

    def get_current_record(
        self,
    ) -> dict[str, Any]:
        """
        Return the current accumulated Schema V2 record.
        """

        return self.session.get_current_record()

    def request_stop(self) -> None:
        """
        Ask continuous microphone capture to stop.
        """

        self.recorder.request_stop()

    def close(self) -> None:
        """
        Close the MongoDB client.
        """

        self.session.close()