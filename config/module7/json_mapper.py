import copy
import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schema_v2.json"
)


def load_schema(
    schema_path: Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """
    Load a clean copy of Schema V2 from its JSON file.
    """

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}"
        )

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def set_value_by_path(
    schema: dict[str, Any],
    schema_path: str,
    value: Any,
) -> None:
    """
    Insert one value into a nested schema path.

    Example:
        schema_path:
            fluids_and_blood.estimated_blood_loss_ml

        value:
            50.0
    """

    path_parts = [
        part.strip()
        for part in schema_path.split(".")
        if part.strip()
    ]

    if not path_parts:
        raise ValueError(
            f"Invalid schema path: '{schema_path}'."
        )

    current_location = schema

    # Navigate through all parts except the final field.
    for part in path_parts[:-1]:
        if part not in current_location:
            raise KeyError(
                f"Schema section '{part}' does not exist "
                f"in path '{schema_path}'."
            )

        next_location = current_location[part]

        if not isinstance(next_location, dict):
            raise TypeError(
                f"Schema path '{part}' is not an object."
            )

        current_location = next_location

    final_field = path_parts[-1]

    if final_field not in current_location:
        raise KeyError(
            f"Schema field '{final_field}' does not exist "
            f"in path '{schema_path}'."
        )

    existing_value = current_location[final_field]

    # Merge list values if the same list field is detected
    # more than once.
    if (
        isinstance(existing_value, list)
        and isinstance(value, list)
    ):
        for item in value:
            if item not in existing_value:
                existing_value.append(
                    copy.deepcopy(item)
                )

    else:
        current_location[final_field] = (
            copy.deepcopy(value)
        )


def map_extraction_results(
    schema: dict[str, Any],
    extraction_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Map accepted extraction results into Schema V2.

    Rejected semantic or value results are preserved in
    the mapping log but are not inserted into the schema.
    """

    mapped_schema = copy.deepcopy(schema)
    mapping_log = []
    mapped_count = 0

    for result in extraction_results:
        status = result.get("status")
        field_name = result.get("field")
        clause = result.get("clause")

        # Only fully accepted results may enter Schema V2.
        if status != "accepted":
            mapping_log.append(
                {
                    "field": field_name,
                    "clause": clause,
                    "status": "skipped",
                    "reason": (
                        f"Extraction status was '{status}'."
                    ),
                }
            )

            continue

        schema_path = result.get("schema_path")
        value = result.get("value")

        if not schema_path:
            mapping_log.append(
                {
                    "field": field_name,
                    "clause": clause,
                    "status": "mapping_error",
                    "reason": (
                        "The accepted result has no "
                        "schema path."
                    ),
                }
            )

            continue

        if value is None:
            mapping_log.append(
                {
                    "field": field_name,
                    "clause": clause,
                    "status": "mapping_error",
                    "reason": (
                        "The accepted result has no value."
                    ),
                }
            )

            continue

        try:
            set_value_by_path(
                schema=mapped_schema,
                schema_path=schema_path,
                value=value,
            )

        except (KeyError, TypeError, ValueError) as error:
            mapping_log.append(
                {
                    "field": field_name,
                    "clause": clause,
                    "status": "mapping_error",
                    "reason": str(error),
                }
            )

            continue

        mapped_count += 1

        mapping_log.append(
            {
                "field": field_name,
                "clause": clause,
                "schema_path": schema_path,
                "value": value,
                "status": "mapped",
                "reason": (
                    "Accepted value mapped successfully."
                ),
            }
        )

    return {
        "schema": mapped_schema,
        "mapping_log": mapping_log,
        "mapped_count": mapped_count,
        "total_results": len(extraction_results),
    }


def save_schema(
    schema: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save the populated Schema V2 record as formatted JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            indent=4,
            ensure_ascii=False,
        )