import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


class SemanticFieldDetectorV2:
    """
    Detects which clinical field an input statement belongs to by
    comparing its embedding with field descriptions and examples.
    """

    def __init__(
        self,
        model_name="FremyCompany/BioLORD-2023-C",
        fields_path=None,
        threshold=0.26,
    ):
        """
        When the detector is created, this constructor:

        1. Selects the GPU or CPU.
        2. Locates semantic_fields.json.
        3. Loads the field descriptions and examples.
        4. Loads the Sentence Transformer model.
        5. Builds the reference bank.
        6. Converts all reference texts into embeddings.

        It does not yet classify an input statement. Classification
        happens later when predict() is called.
        """

        self.model_name = model_name
        self.threshold = threshold

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if fields_path is None:
            fields_path = Path(__file__).with_name(
                "semantic_fields_v2.json"
            )

        self.fields_path = Path(fields_path)

        with self.fields_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.fields = json.load(file)

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        # Stores every unique field name once.
        self.field_names = []

        # Stores every description and example used for comparison.
        self.reference_texts = []

        # Stores the field belonging to each reference text.
        self.reference_fields = []

        # Build the reference bank.
        for field_info in self.fields:
            field_name = field_info["field"]
            description = field_info["description"]
            examples = field_info["examples"]

            self.field_names.append(field_name)

            # Add the field description as one reference.
            self.reference_texts.append(description)
            self.reference_fields.append(field_name)

            # Add each example as another reference.
            for example in examples:
                self.reference_texts.append(example)
                self.reference_fields.append(field_name)

        # Temporary diagnostic for unexpected_events.
        # These lines must be outside both loops.


        # Convert all reference sentences into embeddings.
        self.reference_embeddings = self.model.encode(
            self.reference_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        print("Device:", self.device)
        print("Model:", self.model_name)
        print("Number of fields:", len(self.field_names))
        print(
            "Number of references:",
            len(self.reference_texts),
        )
        print(
            "Embedding shape:",
            self.reference_embeddings.shape,
        )

    def predict(
        self,
        text: str,
        top_k: int = 3,
    ) -> dict:
        """
        Predicts the clinical field associated with an input statement.

        For each field:

        1. Find all references belonging to that field.
        2. Collect their similarity scores.
        3. Calculate the mean similarity score.
        4. Remember the best individual reference.
        5. Sort all fields from highest to lowest score.
        6. Apply the acceptance threshold.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "The input text must be a non-empty string."
            )

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(
                "top_k must be a positive integer."
            )

        # Convert the clinical input statement into an embedding.
        input_embedding = self.model.encode(
            text.strip(),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compare the input with every reference embedding.
        reference_scores = util.cos_sim(
            input_embedding,
            self.reference_embeddings,
        )[0]

        field_results = []

        # Calculate one aggregated result for each field.
        for field_name in self.field_names:
            reference_indices = [
                index
                for index, reference_field
                in enumerate(self.reference_fields)
                if reference_field == field_name
            ]

            scores_for_field = reference_scores[
                reference_indices
            ]

            # Average the reference scores belonging to this field.
            field_score = scores_for_field.mean().item()

            # Find the best individual reference for this field.
            best_local_position = torch.argmax(
                scores_for_field
            ).item()

            best_reference_index = reference_indices[
                best_local_position
            ]

            field_results.append(
                {
                    "field": field_name,
                    "score": field_score,
                    "matched_reference": (
                        self.reference_texts[
                            best_reference_index
                        ]
                    ),
                }
            )

        # Sort the fields from highest score to lowest score.
        field_results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        # The first result is the highest-scoring candidate.
        best_result = field_results[0]

        # Accept it only if it reaches the threshold.
        accepted = (
            best_result["score"] >= self.threshold
        )

        # Return the top-k field candidates.
        top_predictions = [
            {
                "field": result["field"],
                "score": round(result["score"], 4),
                "matched_reference": (
                    result["matched_reference"]
                ),
            }
            for result in field_results[:top_k]
        ]

        return {
            "predicted_field": (
                best_result["field"]
                if accepted
                else None
            ),
            "candidate_field": best_result["field"],
            "score": round(best_result["score"], 4),
            "accepted": accepted,
            "matched_reference": (
                best_result["matched_reference"]
            ),
            "top_predictions": top_predictions,
        }


if __name__ == "__main__":
    detector = SemanticFieldDetectorV2()

    test_statement = (
        "A complication occurred: "
        "temporary oxygen desaturation."
    )

    result = detector.predict(
        test_statement,
        top_k=3,
    )

    print("\nInput:", test_statement)
    print("Predicted field:", result["predicted_field"])
    print("Candidate field:", result["candidate_field"])
    print("Score:", result["score"])
    print("Accepted:", result["accepted"])
    print("Matched reference:", result["matched_reference"])

    print("\nTop predictions:")

    for prediction in result["top_predictions"]:
        print(
            prediction["field"],
            prediction["score"],
            "->",
            prediction["matched_reference"],
        )