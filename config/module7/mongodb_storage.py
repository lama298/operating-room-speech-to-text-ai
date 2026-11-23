import copy
from typing import Any

from pymongo import MongoClient


class MongoDBStorage:
    """
    Stores and retrieves structured surgical records
    in a local MongoDB database.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database_name: str = "or_clinical_documentation",
        collection_name: str = "surgical_records",
    ):
        self.client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )

        self.database = self.client[
            database_name
        ]

        self.collection = self.database[
            collection_name
        ]

    def ping(self) -> bool:
        """
        Verify that MongoDB is reachable.
        """

        result = self.client.admin.command(
            "ping"
        )

        return result.get("ok") == 1.0

    def save_record(
        self,
        record: dict[str, Any],
        record_id: str,
    ) -> dict[str, Any]:
        """
        Insert a new record or replace the existing record
        with the same ID.

        Using replace_one with upsert=True prevents duplicate
        prototype records when the test is run repeatedly.
        """

        document = copy.deepcopy(record)
        document["_id"] = record_id

        result = self.collection.replace_one(
            {"_id": record_id},
            document,
            upsert=True,
        )

        return {
            "record_id": record_id,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": result.upserted_id,
        }

    def get_record(
        self,
        record_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve one record using its MongoDB ID.
        """

        return self.collection.find_one(
            {"_id": record_id}
        )
    def mark_report_generated(
        self,
        record_id: str,
    ) -> dict[str, Any]:


        result = self.collection.update_one(
            {"_id": record_id},
            {
                "$set": {
                    "processing_metadata.report_generated": True
                }
            },
        )

        return {
            "record_id": record_id,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
        }

    def close(self) -> None:
        """
        Close the MongoDB connection.
        """

        self.client.close()