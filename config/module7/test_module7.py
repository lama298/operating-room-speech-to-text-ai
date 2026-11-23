from pathlib import Path

from extractor import load_field_sources, extract_and_map
from json_mapper import load_schema


CONFIG_DIRECTORY = Path(__file__).resolve().parent.parent

FIELD_SOURCES_PATH = (
    CONFIG_DIRECTORY / "field_sources (1).csv"
)

SCHEMA_PATH = CONFIG_DIRECTORY / "schema.json"


schema = load_schema(SCHEMA_PATH)

field_sources = load_field_sources(
    str(FIELD_SOURCES_PATH)
)


test_transcripts = [
    "Primary surgeon is Dr. Lama Mansour",
    "Estimated blood loss is 120",
    "Allergy status confirmed is yes"
]


for transcript in test_transcripts:
    schema, result = extract_and_map(
        transcript,
        field_sources,
        schema
    )

    print("\nTranscript:", transcript)

    if result is None:
        print("No field or value extracted.")

    else:
        print("Detected field:", result["field_name"])
        print("Matched keyword:", result["matched_keyword"])
        print("Extracted value:", result["extracted_value"])
        print("Converted value:", result["converted_value"])
        print("Converted type:", result["converted_type"])


print("\nFinal structured values:")

print(
    "Primary surgeon:",
    schema["surgical_team"]["surgeon_1"]
)

print(
    "Estimated blood loss:",
    schema["fluids_and_blood"]["estimated_blood_loss_ml"]
)

print(
    "Allergy status confirmed:",
    schema["patient_information"]["allergy_status_confirmed"]
)