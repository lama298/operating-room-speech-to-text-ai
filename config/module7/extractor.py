import csv #reads field_sources.csv
from pathlib import Path #handles file path
from typing import Dict, List #descriibe expected data structure
from value_parser import extract_value,convert_value,extract_semantic_value,convert_value_by_type
from json_mapper import map_value
from semantic_detector import SemanticFieldDetector
def load_field_sources(csv_path: str) -> Dict[str, List[str]]:
    """
    Loads all fields and their keywords from field_sources.csv
    """
    path=Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"field_sources.csv was not found: {path.resolve()}"
        )
    field_sources:Dict[str, List[str]]={}
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader= csv.DictReader(csv_file) #without DictReader python reads raw text
#reader now saves rows of the fields
        for row in reader:
            field_name= row["field"].strip() #read the field and take it and store in field name
            keyword_text=row["keywords"].strip() #read keywords and extract entire keywords
            keywords = [ #this makes all keywords each on a line alone  not together
                keyword.strip().lower()
                for keyword in keyword_text.split(",")
                if keyword.strip() #strip removes spaces
            ]
            field_sources[field_name]= keywords
            #at first field_sources was empty, then first iteration it extarcts field nam
            #and keywords, then it puts them all together in the form mentioned above
    return field_sources


def load_field_metadata(
    csv_path: str,
) -> Dict[str, Dict[str, str]]:
    """Loads parser metadata for configured fields."""
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"field_sources.csv was not found: {path.resolve()}"
        )

    field_metadata = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            field_name = (row.get("field") or "").strip()

            parser_strategy = (
                row.get("parser_strategy") or ""
            ).strip()

            data_type = (
                row.get("data_type") or ""
            ).strip()

            unit = (
                row.get("unit") or ""
            ).strip()

            allowed_values = (
                row.get("allowed_values") or ""
            ).strip()

            if not field_name or not parser_strategy:
                continue

            field_metadata[field_name] = {
                "parser_strategy": parser_strategy,
                "data_type": data_type,
                "unit": unit,
                "allowed_values": allowed_values,
            }

    return field_metadata


def detect_field(
        transcript: str,
        field_sources:Dict[str,List[str]]
)-> tuple[str,str]| None: #none means that it didnt find matching field
    """
    detects which structured field is mentioned in a transcript
    """

    normalized_transcript = transcript.strip().lower()
    for field_name, keywords in field_sources.items(): #first loop selects the field
        for keyword in keywords: #second loop checks each keyword separatly to then check whether keyword appears in the transcript
            if keyword in normalized_transcript:
                return field_name,keyword
    return None

def extract_and_map(
        transcript: str,
        field_sources: Dict[str,List[str]],
        schema:dict
):
    """
    Detects the field, extracts and converts its value
    then stores it in the structured schema
    """

    detection_result= detect_field(
        transcript,
        field_sources
    )

    if detection_result is None:
        return schema,  None

    field_name,matched_keyword=detection_result

    extracted_value=extract_value(
        transcript  ,
        matched_keyword
    )

    if extracted_value is None:
        return schema,None

    converted_value=convert_value(extracted_value)

    updated_schema=map_value(
        schema,
        field_name,
        converted_value
    )

    result={
        "transcript":transcript,
        "field_name": field_name,
        "matched_keyword": matched_keyword,
        "extracted_value": extracted_value,
        "converted_value": converted_value,
        "converted_type": type(converted_value).__name__
    }
    return updated_schema, result

def extract_and_map_semantic(
    transcript: str,
    predicted_field: str,
    field_metadata: Dict[str, Dict[str, str]],
    schema: dict,
):
    """
    Extracts, converts, and maps a value for the field
    predicted by BioLORD.
    """
    metadata = field_metadata.get(predicted_field)

    if metadata is None:
        return schema, None

    extracted_value = extract_semantic_value(
        transcript=transcript,
        metadata=metadata,
    )

    if extracted_value is None:
        return schema, None

    try:
        converted_value = convert_value_by_type(
            value=extracted_value,
            data_type=metadata["data_type"],
        )
    except (ValueError, TypeError, KeyError):
        return schema, None

    updated_schema = map_value(
        schema=schema,
        field_name=predicted_field,
        value=converted_value,
    )

    result = {
        "transcript": transcript,
        "predicted_field": predicted_field,
        "parser_strategy": metadata["parser_strategy"],
        "extracted_value": extracted_value,
        "converted_value": converted_value,
        "converted_type": type(converted_value).__name__,
    }

    return updated_schema, result

def predict_extract_and_map(
        transcript: str,
        detector: SemanticFieldDetector,
        field_metadata:Dict[str,Dict[str,str]],
        schema: dict,
):
    """
    Predicts the field with BioLORD, extracts its value,
    converts it, and maps it into the JSON schema
    """
    prediction= detector.predict(
        text= transcript,
        top_k=3,
    )

    predicted_field= prediction["predicted_field"]

    if predicted_field is None:
        result = {
            "status": "rejected",
            "transcript": transcript,
            "prediction": prediction,
            "extraction": None,
        }
        return schema,result

    updated_schema,extraction_result=extract_and_map_semantic(
        transcript=transcript,
        predicted_field=predicted_field,
        field_metadata=field_metadata,
        schema= schema,
    )

    if extraction_result is None:
        result={
            "status":"value_not_extracted",
            "transcript":transcript,
            "prediction":prediction,
            "extraction":None
        }
        return schema, result

    result={
        "status":"mapped",
        "transcript": transcript,
        "prediction":prediction,
        "extraction": extraction_result,
    }

    return updated_schema,result

"""
Conceptually, extractor.py will now coordinate the complete pipeline:
Transcript
→ detect_field()
→ extract_value()
→ convert_value()
→ map_value()
→ updated schema

It does not replace value_parser.py or json_mapper.py; it calls them and coordinates their results.
"""
if __name__ == "__main__":
    csv_path = Path(__file__).with_name(
        "field_sources.csv"
    )

    field_metadata = load_field_metadata(
        str(csv_path)
    )

    test_schema = {
        "intraoperative_events": {
            "unexpected_events": None,
        }
    }

    detector = SemanticFieldDetector()

    transcript = (
        "An unexpected event occurred: equipment malfunction."
    )

    updated_schema, result = predict_extract_and_map(
        transcript=transcript,
        detector=detector,
        field_metadata=field_metadata,
        schema=test_schema,
    )

    prediction = result["prediction"]

    print(
        "Candidate field:",
        prediction["candidate_field"],
    )
    print(
        "Predicted field:",
        prediction["predicted_field"],
    )
    print(
        "Similarity score:",
        prediction["score"],
    )
    print(
        "Accepted:",
        prediction["accepted"],
    )
    print(
        "Pipeline status:",
        result["status"],
    )
    print(
        "Extraction result:",
        result["extraction"],
    )
    print(
        "Stored JSON value:",
        updated_schema["intraoperative_events"][
            "unexpected_events"
        ],
    )