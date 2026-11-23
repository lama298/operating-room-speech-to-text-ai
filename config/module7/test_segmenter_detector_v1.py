from pathlib import Path

from clause_segmenter_v1 import ClauseSegmenterV1
from semantic_detector_v2 import SemanticFieldDetectorV2


class SegmentationDetectionPipeline:
    """
    Connects automatic clause segmentation with the BioLORD
    semantic field detector.

    One transcript may contain several clinical facts.
    The segmenter separates them, and the detector classifies
    every clause independently.
    """

    def __init__(self):
        """
        Create the segmenter and load the semantic detector once.
        """

        module_directory = Path(__file__).parent

        fields_path = (
            module_directory
            / "semantic_fields_v2_selective_refined_v2_79.22.json"
        )

        self.segmenter = ClauseSegmenterV1()

        self.detector = SemanticFieldDetectorV2(
            fields_path=fields_path,
            threshold=0.28,
        )

    def process_transcript(self, transcript, top_k=10):
        """
        Segment a transcript and classify every resulting clause.

        Returns:
            list[dict]: One semantic result for every clause.
        """

        clauses = self.segmenter.segment(
            transcript
        )

        results = []

        for clause in clauses:
            detection = self.detector.predict(
                clause,
                top_k=top_k,
            )

            results.append(
                {
                    "clause": clause,
                    "predicted_field": detection[
                        "predicted_field"
                    ],
                    "candidate_field": detection[
                        "candidate_field"
                    ],
                    "score": detection["score"],
                    "accepted": detection["accepted"],
                    "matched_reference": detection[
                        "matched_reference"
                    ],
                    "top_predictions": detection[
                        "top_predictions"
                    ],
                }
            )

        return results


def print_results(transcript, results):
    """
    Print the segmentation and detection results clearly.
    """

    print("\n" + "=" * 70)
    print("TRANSCRIPT")
    print(transcript)

    print("\nDETECTED CLAUSES")

    for index, result in enumerate(
        results,
        start=1,
    ):
        predicted_field = (
            result["predicted_field"]
            if result["predicted_field"] is not None
            else "REJECTED"
        )

        print(f"\nClause {index}:")
        print("  Text:", result["clause"])
        print("  Predicted field:", predicted_field)
        print(
            "  Candidate field:",
            result["candidate_field"],
        )
        print("  Score:", result["score"])
        print("  Accepted:", result["accepted"])
        print(
            "  Matched reference:",
            result["matched_reference"],
        )

        print("  Top predictions:")

        for prediction in result[
            "top_predictions"
        ]:
            print(
                "   -",
                prediction["field"],
                prediction["score"],
            )


if __name__ == "__main__":
    pipeline = SegmentationDetectionPipeline()

    test_transcripts = [
    (
        "The final counts were correct, "
        "the procedure was completed successfully, "
        "and the patient remained stable."
    ),
]




    for transcript in test_transcripts:
        transcript_results = (
            pipeline.process_transcript(
                transcript,
                top_k=10,
            )
        )

        print_results(
            transcript,
            transcript_results,
        )