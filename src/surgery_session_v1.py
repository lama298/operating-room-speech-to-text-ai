from datetime import datetime, timezone,timedelta
from pathlib import Path
from typing import Any

from config.module7.json_mapper import (
    load_schema,
    map_extraction_results,
    save_schema,
)
from src.end_to_end_pipeline_v1 import (
    EndToEndClinicalPipelineV1,
)
from config.module7.generate_clinical_report_v1 import (generate_html_report)


UTC_LEB=timezone(timedelta(hours=3))
class SurgerySessionV1:
    """
    Accumulates multiple audio segments into one
    structured surgical case.

    The existing end-to-end pipeline components are reused,
    but Schema V2 is loaded only once for the entire case.
    """

    def __init__(
        self,
        record_id: str,
        whisper_model_name: str = "turbo",
        language: str = "en",
        mongo_uri: str = "mongodb://localhost:27017",
        mongo_database: str = (
            "or_clinical_documentation"
        ),
        mongo_collection: str = "surgical_records",
    ):
        """
        Start one empty surgical documentation session.
        """

        if not record_id.strip():
            raise ValueError(
                "record_id cannot be empty."
            )

        self.record_id = record_id.strip()

        # Reuse the already evaluated preprocessing,
        # Whisper, extraction, and MongoDB components.
        self.pipeline = EndToEndClinicalPipelineV1(
            whisper_model_name=whisper_model_name,
            language=language,
            mongo_uri=mongo_uri,
            mongo_database=mongo_database,
            mongo_collection=mongo_collection,
        )

        # This schema is loaded once and then reused for
        # every audio segment in the surgical case.
        self.structured_record = load_schema()

        self.session_started_at = (
            datetime.now(UTC_LEB).isoformat()
        )

        self.utterance_history: list[
            dict[str, Any]
        ] = [] #stores everything about audio segments

        self.extraction_history: list[
            dict[str, Any] #preserves parser and semnatic results
        ] = []

        self.mapping_history: list[
            dict[str, Any]
        ] = [] #stores mapping results like fields and paths

        self.accepted_field_count = 0 #total accepted extraction results
        self.segment_count = 0 #number of processed audio segments
        self.finalized = False #whether surgical case was closed

        self._initialize_session_metadata() #private method to prepare case level metadata

    def _initialize_session_metadata(self) -> None:
        """
        Add case-level session metadata to Schema V2.
        """

        metadata = self.structured_record.get(
            "processing_metadata"
        )

        if not isinstance(metadata, dict):
            metadata = {}
            self.structured_record[
                "processing_metadata"
            ] = metadata

        metadata["session_id"] = self.record_id

        metadata["session_started_at"] = (
            self.session_started_at
        )

        metadata["session_status"] = "active"
        metadata["utterance_count"] = 0
        metadata["accepted_field_count"] = 0
        metadata["utterances"] = []
        metadata["combined_transcript"] = ""
        metadata["report_generated"] = False

        metadata["asr_model"] = (
            self.pipeline.whisper_model_name
        )

        metadata["semantic_model"] = (
            self.pipeline.extraction_pipeline
            .detector.model_name
        )

    def _get_accepted_results(
        self,
        extraction_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return only extraction results accepted by
        field-specific value validation.
        """

        return [
            result
            for result in extraction_results
            if result.get("status") == "accepted"
        ]

    def _update_session_metadata(
        self,
        utterance_entry: dict[str, Any],
    ) -> None:
        """
        Update the accumulated case metadata after one
        audio segment has been processed.
        """

        metadata = self.structured_record[
            "processing_metadata"
        ]

        metadata["utterances"].append(
            utterance_entry
        )

        metadata["utterance_count"] = len(
            self.utterance_history
        )

        metadata["accepted_field_count"] = (
            self.accepted_field_count
        )

        metadata["combined_transcript"] = " ".join(
            entry["transcript"]
            for entry in self.utterance_history
            if entry["transcript"]
        )

        metadata["last_updated_at"] = (
            datetime.now(UTC_LEB).isoformat()
        )

        # Preserve compatibility with the existing
        # single-audio processing metadata.
        metadata["raw_transcript"] = metadata[
            "combined_transcript"
        ]

        metadata["source_audio"] = (
            utterance_entry["source_audio"]
        )

        metadata["processed_at"] = metadata[
            "last_updated_at"
        ]

    def process_audio_segment(
        self,
        input_audio_path: Path | str,
        processed_audio_path: Path | str,
        top_k: int = 3,
        persist: bool = True,
    ) -> dict[str, Any]:
        """
        Process one speech segment and merge its accepted
        values into the current surgical case.
        """

        if self.finalized:
            raise RuntimeError(
                "The surgical session has already been "
                "finalized."
            )

        input_audio_path = Path(
            input_audio_path
        )

        processed_audio_path = Path(
            processed_audio_path
        )

        self.segment_count += 1

        # Stage 1: preprocess only this audio segment.
        preprocessing_result = (
            self.pipeline.preprocessor.preprocess(
                input_path=input_audio_path,
                output_path=processed_audio_path,
            )
        )

        # Stage 2: transcribe only this segment.
        transcription_result = (
            self.pipeline.transcriber.transcribe(
                processed_audio_path
            )
        )

        transcript = transcription_result.get(
            "transcript",
            "",
        ).strip()

        # Stage 3: detect and validate every clinical
        # field present in this segment.
        extraction_results = (
            self.pipeline.extraction_pipeline
            .process_transcript(
                transcript=transcript,
                top_k=top_k,
            )
        )

        accepted_results = (
            self._get_accepted_results(
                extraction_results
            )
        )

        # Stage 4: map into the existing case schema.
        #
        # This is the essential difference from
        # process_audio(), which loads a clean schema for
        # every independent audio file.
        mapping_result = map_extraction_results(
            schema=self.structured_record,
            extraction_results=extraction_results,
        )

        self.structured_record = mapping_result[
            "schema"
        ]

        accepted_fields = [
            result["field"]
            for result in accepted_results
        ]

        utterance_entry = {
            "sequence": self.segment_count,
            "source_audio": input_audio_path.name,
            "processed_audio": (
                processed_audio_path.name
            ),
            "transcript": transcript,
            "accepted_fields": accepted_fields,
            "accepted_field_count": len(
                accepted_results
            ),
            "processed_at": (
                datetime.now(
                    UTC_LEB
                ).isoformat()
            ),
        }

        self.utterance_history.append(
            utterance_entry
        )

        self.extraction_history.extend(
            extraction_results
        )

        self.mapping_history.extend(
            mapping_result["mapping_log"]
        )

        self.accepted_field_count += len(
            accepted_results
        )

        self._update_session_metadata(
            utterance_entry
        )

        mongo_save_result = None

        # Do not create a MongoDB record until at least one
        # validated clinical field has been captured.
        if (
            persist
            and self.accepted_field_count > 0
        ):
            if not self.pipeline.storage.ping():
                raise ConnectionError(
                    "MongoDB did not respond "
                    "successfully."
                )

            mongo_save_result = (
                self.pipeline.storage.save_record(
                    record=self.structured_record,
                    record_id=self.record_id,
                )
            )

        return {
            "record_id": self.record_id,
            "segment_number": self.segment_count,
            "input_audio_path": str(
                input_audio_path
            ),
            "processed_audio_path": str(
                processed_audio_path
            ),
            "preprocessing": preprocessing_result,
            "transcription": transcription_result,
            "transcript": transcript,
            "extractions": extraction_results,
            "accepted_extractions": (
                accepted_results
            ),
            "mapping": mapping_result,
            "structured_record": (
                self.structured_record
            ),
            "mongo_save_result": (
                mongo_save_result
            ),
        }

    def get_current_record(
        self,
    ) -> dict[str, Any]:
        """
        Return the current in-memory surgical record.
        """

        return self.structured_record

    def save_current_record(
        self,
    ) -> dict[str, Any] | None:
        """
        Save the complete accumulated case to MongoDB.
        """

        if self.accepted_field_count == 0:
            return None

        if not self.pipeline.storage.ping():
            raise ConnectionError(
                "MongoDB did not respond successfully."
            )

        return self.pipeline.storage.save_record(
            record=self.structured_record,
            record_id=self.record_id,
        )

    def finalize_case(
        self,
        schema_output_path: Path | str | None = None,
        report_output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Finalize the complete surgical case.

        The accumulated record is saved to MongoDB before
        the deterministic HTML report is generated.
        """

        if self.finalized:
            raise RuntimeError(
                "The surgical session has already been "
                "finalized."
            )

        if self.accepted_field_count == 0:
            raise ValueError(
                "The session contains no accepted "
                "clinical fields."
            )

        finalized_at = (
            datetime.now(UTC_LEB).isoformat()
        )

        metadata = self.structured_record[
            "processing_metadata"
        ]

        metadata["session_status"] = "finalized"
        metadata["session_finalized_at"] = (
            finalized_at
        )

        # Save the complete accumulated record first.
        mongo_save_result = (
            self.save_current_record()
        )

        report_path = None
        report_status_result = None

        if report_output_path is not None:
            report_output_path = Path(
                report_output_path
            )

            stored_record = (
                self.pipeline.storage.get_record(
                    self.record_id
                )
            )

            if stored_record is None:
                raise AssertionError(
                    "The accumulated MongoDB record "
                    "could not be retrieved for report "
                    "generation."
                )

            html_report = generate_html_report(
                stored_record
            )

            report_output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            report_output_path.write_text(
                html_report,
                encoding="utf-8",
            )

            report_path = str(
                report_output_path
            )

            # Update MongoDB only after the HTML file
            # has been written successfully.
            report_status_result = (
                self.pipeline.storage
                .mark_report_generated(
                    self.record_id
                )
            )

            # Keep the in-memory record synchronized.
            metadata["report_generated"] = True

        saved_schema_path = None

        # Save JSON after report generation so the JSON
        # contains the final report status.
        if schema_output_path is not None:
            schema_output_path = Path(
                schema_output_path
            )

            save_schema(
                schema=self.structured_record,
                output_path=schema_output_path,
            )

            saved_schema_path = str(
                schema_output_path
            )

        self.finalized = True

        return {
            "record_id": self.record_id,
            "segment_count": self.segment_count,
            "accepted_field_count": (
                self.accepted_field_count
            ),
            "structured_record": (
                self.structured_record
            ),
            "saved_schema_path": (
                saved_schema_path
            ),
            "report_output_path": report_path,
            "report_status_result": (
                report_status_result
            ),
            "mongo_save_result": (
                mongo_save_result
            ),
            "finalized_at": finalized_at,
        }

    def get_stored_record(
        self,
    ) -> dict[str, Any] | None:
        """
        Retrieve the accumulated case from MongoDB.
        """

        return self.pipeline.storage.get_record(
            self.record_id
        )

    def close(self) -> None:
        """
        Close the shared MongoDB client.
        """

        self.pipeline.close()