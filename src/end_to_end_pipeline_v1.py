from datetime import datetime, timezone,timedelta
from pathlib import Path
from typing import Any

from src.audio_processing.audio_preprocessor_v1 import (
    AudioPreprocessorV1,
)
from src.audio_processing.whisper_transcriber_v1 import (
    WhisperTranscriberV1,
)
from config.module7.clinical_extraction_pipeline_v1 import (
    ClinicalExtractionPipelineV1,
)
from config.module7.json_mapper import (
    load_schema,
    map_extraction_results,
    save_schema,
)
from config.module7.mongodb_storage import (
    MongoDBStorage,
)


UTC_LEB=timezone(timedelta(hours=3))
class EndToEndClinicalPipelineV1:
    """
    Coordinates the complete audio-to-database workflow:

    1. Preprocess audio.
    2. Transcribe audio using Whisper.
    3. Detect clinical fields and parse their values.
    4. Map accepted values into Schema V2.
    5. Optionally save Schema V2 as a JSON file.
    6. Optionally save Schema V2 to MongoDB.
    """

    def __init__(
        self,
        whisper_model_name: str = "turbo",
        language: str = "en",
        mongo_uri: str = "mongodb://localhost:27017",
        mongo_database: str = (
            "or_clinical_documentation"
        ),
        mongo_collection: str = "surgical_records",
    ):
        """
        Initialize reusable pipeline and database components.
        """

        self.whisper_model_name = whisper_model_name

        self.preprocessor = AudioPreprocessorV1()

        self.transcriber = WhisperTranscriberV1(
            model_name=whisper_model_name,
            language=language,
        )

        self.extraction_pipeline = (
            ClinicalExtractionPipelineV1()
        )

        self.storage = MongoDBStorage(
            uri=mongo_uri,
            database_name=mongo_database,
            collection_name=mongo_collection,
        )

    def add_processing_metadata(
        self,
        structured_record: dict[str, Any],
        input_audio_path: Path,
        transcript: str,
    ) -> None:
        """
        Add traceability information to Schema V2.
        """

        metadata = structured_record.get(
            "processing_metadata"
        )

        if not isinstance(metadata, dict):
            return

        metadata["source_audio"] = input_audio_path.name
        metadata["raw_transcript"] = transcript
        metadata["asr_model"] = self.whisper_model_name

        metadata["semantic_model"] = (
            self.extraction_pipeline.detector.model_name
        )

        metadata["processed_at"] = (
            datetime.now(UTC_LEB).isoformat()
        )

    def process_audio(
        self,
        input_audio_path: Path | str,
        processed_audio_path: Path | str,
        schema_output_path: Path | str | None = None,
        record_id: str | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Process one audio file through the complete pipeline.

        Providing record_id enables MongoDB storage.
        """

        input_audio_path = Path(input_audio_path)

        processed_audio_path = Path(
            processed_audio_path
        )

        # Stage 1: Audio preprocessing.
        preprocessing_result = (
            self.preprocessor.preprocess(
                input_path=input_audio_path,
                output_path=processed_audio_path,
            )
        )

        # Stage 2: Whisper transcription.
        transcription_result = (
            self.transcriber.transcribe(
                processed_audio_path
            )
        )

        transcript = transcription_result[
            "transcript"
        ]

        # Stage 3: Field detection and value parsing.
        extraction_results = (
            self.extraction_pipeline.process_transcript(
                transcript=transcript,
                top_k=top_k,
            )
        )

        # Stage 4: Map accepted values into Schema V2.
        clean_schema = load_schema()

        mapping_result = map_extraction_results(
            schema=clean_schema,
            extraction_results=extraction_results,
        )

        populated_schema = mapping_result["schema"]

        # Add traceability metadata before storage.
        self.add_processing_metadata(
            structured_record=populated_schema,
            input_audio_path=input_audio_path,
            transcript=transcript,
        )

        # Stage 5: Optional JSON-file storage.
        saved_schema_path = None

        if schema_output_path is not None:
            schema_output_path = Path(
                schema_output_path
            )

            save_schema(
                schema=populated_schema,
                output_path=schema_output_path,
            )

            saved_schema_path = str(
                schema_output_path
            )

        # Stage 6: Optional MongoDB storage.
        mongo_save_result = None

        if record_id is not None:
            if not record_id.strip():
                raise ValueError(
                    "record_id cannot be empty."
                )

            if not self.storage.ping():
                raise ConnectionError(
                    "MongoDB did not respond successfully."
                )

            mongo_save_result = (
                self.storage.save_record(
                    record=populated_schema,
                    record_id=record_id,
                )
            )

        return {
            "input_audio_path": str(input_audio_path),
            "processed_audio_path": str(
                processed_audio_path
            ),
            "preprocessing": preprocessing_result,
            "transcription": transcription_result,
            "transcript": transcript,
            "extractions": extraction_results,
            "mapping": mapping_result,
            "structured_record": populated_schema,
            "saved_schema_path": saved_schema_path,
            "mongo_save_result": mongo_save_result,
        }

    def get_record(
        self,
        record_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve one structured record from MongoDB.
        """

        return self.storage.get_record(record_id)

    def close(self) -> None:
        """
        Close the MongoDB client.
        """

        self.storage.close()