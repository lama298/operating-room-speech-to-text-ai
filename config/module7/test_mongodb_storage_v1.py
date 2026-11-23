import json
from pathlib import Path

from mongodb_storage import MongoDBStorage


if __name__ == "__main__":
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    record_path = (
        project_root
        / "outputs"
        / "structured_case_v1.json"
    )

    with record_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        structured_record = json.load(file)

    storage = MongoDBStorage()

    try:
        print(
            "MongoDB connected:",
            storage.ping(),
        )

        save_result = storage.save_record(
            record=structured_record,
            record_id="prototype_case_v1",
        )

        print("\nSAVE RESULT")
        print(
            json.dumps(
                save_result,
                indent=4,
                default=str,
            )
        )

        retrieved_record = storage.get_record(
            "prototype_case_v1"
        )

        print("\nRETRIEVED RECORD")

        print(
            json.dumps(
                retrieved_record,
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )

        print(
            "\nRetrieval successful:",
            retrieved_record is not None,
        )

    finally:
        storage.close()