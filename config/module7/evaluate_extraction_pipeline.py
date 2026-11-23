import csv
from collections import Counter #counts pipeline status such as mapped and rejected
from pathlib import Path

from extractor import (load_field_metadata #loads each field's parser strategy and data type from field_sources.csv
                       , predict_extract_and_map #runs the complete eextraction pipeline for one transcript
                       )
from json_mapper import load_schema #loads a clean clinical json schema for each case
from semantic_detector import SemanticFieldDetector #loads BIOLORD and predcits which clinical field the senetence represents

BASE_DIR= Path(__file__).resolve().parent #give the location of the evaluator script, resolve converts it into a complete absolute path, and select the directory
CASES_PATH=BASE_DIR / "final_test_eval.csv" #input evaluation datastet
METADATA_PATH= BASE_DIR / "field_sources.csv" #parser configuration for the 12 fields
RESULTS_PATH=BASE_DIR / "evaluation_results2.csv" #detailed results that the evaluator will create

THRESHOLD= 0.26

#compare predicted and expected values
def values_equal(actual, expected: str)-> bool:
    """Compare the extracted value with the GT value"""

    if actual is None: #return false when the parser extarcts nothing
        return False

    expected=expected.strip()

    if isinstance(actual,bool): #compare boolean values such as True and "true"
        return str(actual).lower() == expected.lower()
    try:
        return abs(float(actual) - float(expected)) < 0.0001 #compare numeric values, with floatinf point tolerance
    except (TypeError, ValueError):
        return str(actual).strip().casefold() == expected.casefold() #compare text after removinf extra spaces and ignoring capitalization


#calc %
def percentage(correct: int, total: int)-> float:
    return 100*correct / total if total else 0.0

#load pipelnine and eval dataset

def main()->None:
    field_metadata= load_field_metadata(str(METADATA_PATH)) #load metadata describing each field's parser and data type
    detector= SemanticFieldDetector (threshold= THRESHOLD) #initialize BioLORD once usinig threshold=0.26

    with CASES_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file: cases=list(csv.DictReader(file))

    print(f"Loaded {len(cases)} evaluation cases")

    results = []

    for case in cases:
        should_accept=(case["should_accept"].strip().lower() == "true")
        schema = load_schema()

        _, pipeline_result = predict_extract_and_map(
            transcript=case["transcript"],
            detector=detector,
            field_metadata=field_metadata,
            schema=schema,
        )

        prediction = pipeline_result.get("prediction", {})
        extraction = pipeline_result.get("extraction") or {}
        predicted_field = prediction.get("predicted_field")
        accepted = prediction.get("accepted", False)
        converted_value = extraction.get("converted_value")

        if should_accept:
            field_correct = (
                accepted
                and predicted_field == case["expected_field"]
            )

            value_correct = values_equal(
                converted_value,
                case["expected_value"],
            )

            case_correct = field_correct and value_correct

        else:
            field_correct = False
            value_correct = False
            case_correct = not accepted

        results.append(
            {
                "test_id": case["test_id"],
                "transcript": case["transcript"],
                "expected_field": case["expected_field"],
                "expected_value": case["expected_value"],
                "should_accept": should_accept,
                "candidate_field": prediction.get("candidate_field"),
                "predicted_field": predicted_field,
                "score": prediction.get("score"),
                "accepted": accepted,
                "pipeline_status": pipeline_result.get("status"),
                "converted_value": converted_value,
                "field_correct": field_correct,
                "value_correct": value_correct,
                "case_correct": case_correct,
            }
        )
        positives = [
        result
        for result in results
        if result["should_accept"]
    ]

    negatives = [
        result
        for result in results
        if not result["should_accept"]
    ]
    correct_fields = sum(
        result["field_correct"]
        for result in positives
    )

    correct_values = sum(
        result["value_correct"]
        for result in positives
    )

    correct_positive_cases = sum(
        result["case_correct"]
        for result in positives
    )

    correct_rejections = sum(
        result["case_correct"]
        for result in negatives
    )

    correct_overall = sum(
        result["case_correct"]
        for result in results
    )

    print("Correct fields:", correct_fields)
    print("Correct values:", correct_values)
    print("Correct positive cases:", correct_positive_cases)
    print("Correct rejections:", correct_rejections)
    print("Correct overall:", correct_overall)
    print("\nBASELINE EVALUATION")
    print("-------------------")
    print(f"Threshold: {THRESHOLD}")
    print(f"Total cases: {len(results)}")

    print(
        "Field-detection accuracy:",
        f"{correct_fields}/{len(positives)}",
        f"({percentage(correct_fields, len(positives)):.2f}%)",
    )

    print(
        "Value-extraction accuracy:",
        f"{correct_values}/{len(positives)}",
        f"({percentage(correct_values, len(positives)):.2f}%)",
    )

    print(
        "Positive end-to-end accuracy:",
        f"{correct_positive_cases}/{len(positives)}",
        f"({percentage(correct_positive_cases, len(positives)):.2f}%)",
    )

    print(
        "Rejection accuracy:",
        f"{correct_rejections}/{len(negatives)}",
        f"({percentage(correct_rejections, len(negatives)):.2f}%)",
    )

    print(
        "Overall accuracy:",
        f"{correct_overall}/{len(results)}",
        f"({percentage(correct_overall, len(results)):.2f}%)",
    )
    failed_results = [
        result
        for result in results
        if not result["case_correct"]
    ]

    print("\nFAILED CASES")
    print("------------")
    print(f"Number of failures: {len(failed_results)}")

    for result in failed_results:
        if not result["should_accept"]:
            failure_type = "false acceptance"
        elif not result["field_correct"]:
            failure_type = "field detection"
        else:
            failure_type = "value extraction"

        print(f"\nTest ID: {result['test_id']}")
        print(f"Failure type: {failure_type}")
        print(f"Transcript: {result['transcript']}")
        print(f"Expected field: {result['expected_field']}")
        print(f"Predicted field: {result['predicted_field']}")
        print(f"Expected value: {result['expected_value']}")
        print(f"Converted value: {result['converted_value']}")
        print(f"Score: {result['score']}")
        print(f"Pipeline status: {result['pipeline_status']}")


    print("Positive cases:", len(positives))
    print("Negative cases:", len(negatives))

    print(f"Processed {len(results)} evaluation cases")
    print("First collected result:")
    print(results[0])

if __name__ == "__main__":
    main()