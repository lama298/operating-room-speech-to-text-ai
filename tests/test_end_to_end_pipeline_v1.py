import json
import sys
from pathlib import Path

from src.end_to_end_pipeline_v1 import EndToEndClinicalPipelineV1


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The report module imports mongodb_storage from its own directory.
sys.path.insert(0, str(PROJECT_ROOT / "config" / "module7"))
from config.module7.generate_clinical_report_v1 import generate_html_report


INPUT_AUDIO_PATH = (
    PROJECT_ROOT / "audio" / "pilot" / "raw" / "STD_008.m4a"
)
PROCESSED_AUDIO_PATH = (
    PROJECT_ROOT / "outputs" / "end_to_end_test" / "STD_008_processed.wav"
)
SCHEMA_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "end_to_end_test" / "STD_008_structured.json"
)
RECORD_ID = "STD_008"


def main():
    print("Creating end-to-end pipeline...")
    pipeline = EndToEndClinicalPipelineV1(
        whisper_model_name="turbo",
        language="en",
    )

    try:
        print("Processing audio...")
        result = pipeline.process_audio(
            input_audio_path=INPUT_AUDIO_PATH,
            processed_audio_path=PROCESSED_AUDIO_PATH,
            schema_output_path=SCHEMA_OUTPUT_PATH,
            record_id=RECORD_ID,
            top_k=3,
        )

        print("\nEND-TO-END RESULT")
        print("=" * 70)
        print("Transcript:", result["transcript"])

        print("\nVALIDATED EXTRACTIONS")
        for index, extraction in enumerate(result["extractions"], start=1):
            print(f"\nExtraction {index}")
            print("Clause:", extraction["clause"])
            print("Status:", extraction["status"])
            print("Field:", extraction["field"])
            print("Parser strategy:", extraction["parser_strategy"])
            print("Semantic score:", extraction["semantic_score"])
            print("Value:", extraction["value"])
            print("Reason:", extraction["reason"])

        mapping_result = result["mapping"]
        print("\nMAPPING SUMMARY")
        for entry in mapping_result["mapping_log"]:
            print(json.dumps(entry, indent=4, ensure_ascii=False, default=str))
        print("=" * 70)
        print("Total extraction results:", mapping_result["total_results"])
        print("Successfully mapped:", mapping_result["mapped_count"])

        print("\nMONGODB SAVE RESULT")
        print("=" * 70)
        print(json.dumps(result["mongo_save_result"], indent=4, default=str))

        retrieved_record = pipeline.get_record(RECORD_ID)
        print("\nRETRIEVED MONGODB RECORD")
        print("=" * 70)
        print(
            json.dumps(
                retrieved_record,
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )

        if retrieved_record is None:
            raise AssertionError("MongoDB record was not retrieved.")

        retrieved_outcome = retrieved_record["patient_outcome"]
        print(
            "Retrieved patient outcome:",
            json.dumps(retrieved_outcome, indent=4, ensure_ascii=False),
        )
        assert retrieved_outcome["destination"] == "pacu"
        assert retrieved_outcome["condition"] == "stable"
        print("MongoDB verification passed.")

        # Generate the report from the record retrieved from MongoDB.
        report_html = generate_html_report(retrieved_record)
        report_path = (
            PROJECT_ROOT
            / "outputs"
            / "end_to_end_test"
            / f"clinical_report_{RECORD_ID}.html"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_html, encoding="utf-8")

        # Mark success only after the HTML file has been written.
        update_result = pipeline.storage.mark_report_generated(RECORD_ID)
        assert update_result["matched_count"] == 1
        print("HTML report:", report_path)
        print("Report status update:", update_result)
    finally:
        pipeline.close()
        print("\nMongoDB client closed.")


if __name__ == "__main__":
    main()
