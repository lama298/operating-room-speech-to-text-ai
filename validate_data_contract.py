import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

SCHEMA_PATH = (
    PROJECT_DIR
    / "config"
    / "schema_v2.json"
)

REGISTRY_PATH = (
    PROJECT_DIR
    / "config"
    / "module7"
    / "field_sources_v2.csv"
)


def schema_path_exists(schema, schema_path):
    current_value = schema

    for path_part in schema_path.split("."):
        if not isinstance(current_value, dict):
            return False

        if path_part not in current_value:
            return False

        current_value = current_value[path_part]

    return True


with SCHEMA_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    schema = json.load(file)


with REGISTRY_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    registry_rows = list(csv.DictReader(file))


field_counts = Counter(
    row["field"]
    for row in registry_rows
)

path_counts = Counter(
    row["schema_path"]
    for row in registry_rows
)


duplicate_fields = [
    field
    for field, count in field_counts.items()
    if count > 1
]

duplicate_paths = [
    path
    for path, count in path_counts.items()
    if count > 1
]


missing_schema_paths = [
    row
    for row in registry_rows
    if not schema_path_exists(
        schema,
        row["schema_path"],
    )
]


invalid_detector_rows = []

for row in registry_rows:
    included = (
        row["include_in_detector"]
        .strip()
        .lower()
        == "true"
    )

    if not included:
        continue

    required_columns = [
        "field",
        "schema_path",
        "keywords",
        "parser_strategy",
        "data_type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if not row[column].strip()
    ]

    if missing_columns:
        invalid_detector_rows.append(
            {
                "field": row["field"],
                "missing_columns": missing_columns,
            }
        )


detector_target_count = sum(
    row["include_in_detector"]
    .strip()
    .lower()
    == "true"
    for row in registry_rows
)


print("=" * 60)
print("DATA CONTRACT VALIDATION")
print("=" * 60)
print("Registry rows:", len(registry_rows))
print("Detector targets:", detector_target_count)
print("Duplicate fields:", len(duplicate_fields))
print("Duplicate schema paths:", len(duplicate_paths))
print("Missing schema paths:", len(missing_schema_paths))
print("Invalid detector rows:", len(invalid_detector_rows))


if duplicate_fields:
    print("\nDuplicate fields:")
    for field in duplicate_fields:
        print("-", field)


if duplicate_paths:
    print("\nDuplicate schema paths:")
    for path in duplicate_paths:
        print("-", path)


if missing_schema_paths:
    print("\nMissing schema paths:")
    for row in missing_schema_paths:
        print(
            "-",
            row["field"],
            "->",
            row["schema_path"],
        )


if invalid_detector_rows:
    print("\nInvalid detector rows:")
    for row in invalid_detector_rows:
        print(
            "-",
            row["field"],
            "missing:",
            ", ".join(row["missing_columns"]),
        )


validation_passed = not any(
    [
        duplicate_fields,
        duplicate_paths,
        missing_schema_paths,
        invalid_detector_rows,
    ]
)


print("\nValidation passed:", validation_passed)


if not validation_passed:
    raise SystemExit(1)