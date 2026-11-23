import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PROJECT_DIR / "config" / "schema.json"


def identify_default_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"

    return type(value).__name__


def collect_leaf_fields(data, parent_path=""):
    fields = []

    for key, value in data.items():
        field_path = (
            f"{parent_path}.{key}"
            if parent_path
            else key
        )

        # Recurse only into non-empty dictionaries.
        if isinstance(value, dict) and value:
            fields.extend(
                collect_leaf_fields(
                    value,
                    parent_path=field_path,
                )
            )
        else:
            fields.append(
                {
                    "path": field_path,
                    "default_type": identify_default_type(value),
                }
            )

    return fields


with SCHEMA_PATH.open("r", encoding="utf-8") as file:
    schema = json.load(file)


leaf_fields = collect_leaf_fields(schema)

type_counts = Counter(
    field["default_type"]
    for field in leaf_fields
)


print("=" * 70)
print("RECURSIVE SCHEMA INVENTORY")
print("=" * 70)

for index, field in enumerate(leaf_fields, start=1):
    print(
        f"{index:03d} | "
        f"{field['default_type']:<8} | "
        f"{field['path']}"
    )


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Top-level sections:", len(schema))
print("Recursive leaf fields:", len(leaf_fields))

for default_type, count in sorted(type_counts.items()):
    print(f"{default_type}: {count}")