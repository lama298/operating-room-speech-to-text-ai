import csv
import json
from collections import Counter
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent

REGISTRY_PATH = (
    MODULE_DIR
    / "field_sources_v2.csv"
)

SEMANTIC_FIELDS_PATH = (
    MODULE_DIR
    / "semantic_fields_v2.json"
)


with REGISTRY_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    registry_rows = list(csv.DictReader(file))


with SEMANTIC_FIELDS_PATH.open(
    "r",
    encoding="utf-8-sig",
) as file:
    semantic_fields = json.load(file)


detector_targets = {
    row["field"]
    for row in registry_rows
    if (
        row["include_in_detector"]
        .strip()
        .lower()
        == "true"
    )
}


semantic_names = [
    field_info["field"]
    for field_info in semantic_fields
]

semantic_name_set = set(semantic_names)


name_counts = Counter(semantic_names)

duplicate_names = sorted(
    name
    for name, count in name_counts.items()
    if count > 1
)


unknown_names = sorted(
    semantic_name_set
    - detector_targets
)


missing_names = sorted(
    detector_targets
    - semantic_name_set
)


invalid_definitions = []

for field_info in semantic_fields:
    field_name = field_info.get("field", "")
    description = field_info.get("description", "")
    examples = field_info.get("examples", [])

    problems = []

    if not field_name.strip():
        problems.append("missing field name")

    if not description.strip():
        problems.append("missing description")

    if not isinstance(examples, list):
        problems.append("examples is not a list")
    elif len(examples) < 3:
        problems.append("fewer than 3 examples")

    if problems:
        invalid_definitions.append(
            {
                "field": field_name,
                "problems": problems,
            }
        )


coverage_percentage = (
    len(semantic_name_set & detector_targets)
    / len(detector_targets)
    * 100
)


print("=" * 60)
print("SEMANTIC COVERAGE VALIDATION")
print("=" * 60)
print("Detector targets:", len(detector_targets))
print("Semantic fields:", len(semantic_fields))
print("Covered targets:", len(
    semantic_name_set & detector_targets
))
print("Missing targets:", len(missing_names))
print("Unknown targets:", len(unknown_names))
print("Duplicate targets:", len(duplicate_names))
print("Invalid definitions:", len(invalid_definitions))
print(f"Coverage: {coverage_percentage:.2f}%")


if missing_names:
    print("\nMissing target names:")

    for name in missing_names:
        print("-", name)


if unknown_names:
    print("\nUnknown target names:")

    for name in unknown_names:
        print("-", name)


if duplicate_names:
    print("\nDuplicate target names:")

    for name in duplicate_names:
        print("-", name)


if invalid_definitions:
    print("\nInvalid definitions:")

    for item in invalid_definitions:
        print(
            "-",
            item["field"],
            ":",
            ", ".join(item["problems"]),
        )


configuration_valid = not any(
    [
        unknown_names,
        duplicate_names,
        invalid_definitions,
    ]
)


print(
    "\nCurrent configuration valid:",
    configuration_valid,
)