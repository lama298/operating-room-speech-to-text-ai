import csv
from pathlib import Path
from typing import Any
import re

try:
    # Used when imported as part of config.module7.
    from .clause_segmenter_v1 import (
        ClauseSegmenterV1,
    )
    from .semantic_detector_v2 import (
        SemanticFieldDetectorV2,
    )
    from .value_parser import (
        parse_and_validate,
    )

except ImportError:
    # Allows this file to run directly.
    from clause_segmenter_v1 import (
        ClauseSegmenterV1,
    )
    from semantic_detector_v2 import (
        SemanticFieldDetectorV2,
    )
    from value_parser import (
        parse_and_validate,
    )


class ClinicalExtractionPipelineV1:
    """
    Connects:

    1. Automatic clause segmentation.
    2. BioLORD semantic field detection.
    3. Field-aware value parsing.
    4. Parser-guided semantic fallback.
    5. Parser-confirmed secondary extraction.
    6. Constrained below-threshold event recovery.
    7. Explicit-label parser recovery.
    """

    def __init__(self):
        module_directory = Path(__file__).parent

        fields_path = (
            module_directory
            / (
                "semantic_fields_v2_"
                "selective_refined_v2_79.22.json"
            )
        )

        metadata_path = (
            module_directory
            / "field_sources_v2.csv"
        )

        self.segmenter = ClauseSegmenterV1()

        self.detector = SemanticFieldDetectorV2(
            fields_path=fields_path,
            threshold=0.28,
        )

        self.field_metadata = (
            self.load_field_metadata(
                metadata_path
            )
        )

        # These parsers require identifiable,
        # field-specific evidence in a clause.
        self.strict_secondary_strategies = {
            "structured_list",
            "categorical",
            "categorical_list",
            "count_action_list",
            "person_name",
            "identifier",
            "allergy_list",
            "time",
            "event_text",
            "event_list",
            "boolean",
            "procedure_text",
        }
        self.strict_secondary_measurement_fields = {
            "tourniquet_pressure_mmhg",
            "tourniquet_duration_minutes",
        }
        self.strict_secondary_negative_boolean_fields = {
            "count_discrepancy",
        }

        # A secondary field must retain at least half
        # the primary semantic score.
        self.secondary_score_ratio = 0.50

        # Conservative values used only for structured
        # unexpected-event recovery.
        self.event_rescue_minimum_score = 0.20
        self.event_rescue_relative_ratio = 0.70

    def load_field_metadata(
        self,
        metadata_path: Path,
    ) -> dict[str, dict[str, str]]:
        """
        Load field_sources_v2.csv and index every row
        using its semantic field name.
        """

        with metadata_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = csv.DictReader(file)

            return {
                row["field"]: row
                for row in rows
            }

    def create_extraction_result(
        self,
        clause: str,
        field: str,
        candidate_field: str | None,
        semantic_score: float | None,
        metadata: dict[str, str],
        validation_result: dict[str, Any],
        matched_reference: str | None,
        selection_strategy: str,
    ) -> dict[str, Any]:
        """
        Build one standardized extraction result.
        """

        if validation_result["accepted"]:
            status = "accepted"
        else:
            status = "value_rejected"

        reason = validation_result["reason"]

        if (
            status == "accepted"
            and selection_strategy
            == "secondary_parser_confirmed"
        ):
            reason = (
                "Secondary semantic candidate confirmed "
                "by a strict field-specific parser."
            )

        elif (
            status == "accepted"
            and selection_strategy
            == "parser_guided_fallback"
        ):
            reason = (
                "Primary candidate failed value "
                "validation; an alternative semantic "
                "candidate was confirmed by its parser."
            )

        elif (
            status == "accepted"
            and selection_strategy
            == "parser_guided_event_rescue"
        ):
            reason = (
                "Below-threshold semantic event "
                "candidate confirmed by structured "
                "event parsing."
            )

        elif (
            status == "accepted"
            and selection_strategy
            == "explicit_label_parser_confirmed"
        ):
            reason = (
                "Explicit clinical label confirmed by "
                "strict field-specific parsing."
            )

        return {
            "clause": clause,
            "status": status,
            "field": field,
            "candidate_field": candidate_field,
            "semantic_score": semantic_score,
            "parser_strategy": metadata[
                "parser_strategy"
            ],
            "schema_path": metadata[
                "schema_path"
            ],
            "extracted_value": validation_result[
                "extracted_value"
            ],
            "value": validation_result[
                "converted_value"
            ],
            "reason": reason,
            "matched_reference": matched_reference,
            "selection_strategy": (
                selection_strategy
            ),
        }

    def find_parser_guided_fallback(
        self,
        clause: str,
        semantic_result: dict[str, Any],
        rejected_field: str,
    ) -> dict[str, Any] | None:
        """
        Find an alternative semantic field when the
        primary field fails value validation.

        Permissive text fallback is allowed only when
        the rejected primary field expected a time.
        """

        rejected_metadata = (
            self.field_metadata.get(
                rejected_field
            )
        )

        if rejected_metadata is None:
            return None

        rejected_parser_strategy = (
            rejected_metadata.get(
                "parser_strategy",
                "",
            )
            .strip()
            .lower()
        )

        for prediction in semantic_result.get(
            "top_predictions",
            [],
        ):
            candidate_field = prediction.get(
                "field"
            )

            candidate_score = prediction.get(
                "score",
                0.0,
            )

            if not candidate_field:
                continue

            # Do not retry the rejected primary field.
            if candidate_field == rejected_field:
                continue

            # Normal fallbacks must pass the global
            # semantic threshold.
            if (
                candidate_score
                < self.detector.threshold
            ):
                continue

            candidate_metadata = (
                self.field_metadata.get(
                    candidate_field
                )
            )

            if candidate_metadata is None:
                continue

            candidate_parser_strategy = (
                candidate_metadata.get(
                    "parser_strategy",
                    "",
                )
                .strip()
                .lower()
            )

            # A text parser can accept almost anything.
            # Permit it only when a time field was
            # predicted but no real time was found.
            if (
                candidate_parser_strategy == "text"
                and rejected_parser_strategy != "time"
            ):
                continue

            candidate_validation = (
                parse_and_validate(
                    transcript=clause,
                    metadata=candidate_metadata,
                )
            )

            if not candidate_validation[
                "accepted"
            ]:
                continue

            # A generic negative word such as "no" is
            # unsafe evidence for a fallback Boolean
            # field.
            if (
                candidate_parser_strategy == "boolean"
                and candidate_validation[
                    "converted_value"
                ] is False
            ):
                continue

            return {
                "field": candidate_field,
                "score": candidate_score,
                "matched_reference": (
                    prediction.get(
                        "matched_reference"
                    )
                ),
                "metadata": candidate_metadata,
                "validation": (
                    candidate_validation
                ),
            }

        return None

    def extract_secondary_results(
        self,
        clause: str,
        semantic_result: dict[str, Any],
        excluded_fields: set[str],
    ) -> list[dict[str, Any]]:


        secondary_results = []

        primary_score = semantic_result.get(
            "score",
            0.0,
        )

        minimum_relative_score = (
            primary_score
            * self.secondary_score_ratio
        )

        top_predictions = semantic_result.get(
            "top_predictions",
            [],
        )

        # Determine the parser used by the primary
        # accepted semantic field.
        primary_field = semantic_result.get(
            "predicted_field"
        )

        primary_metadata = self.field_metadata.get(
            primary_field,
            {},
        )

        primary_parser_strategy = (
            primary_metadata.get(
                "parser_strategy",
                "",
            )
            .strip()
            .lower()
        )

        for prediction in top_predictions:
            candidate_field = prediction.get(
                "field"
            )

            candidate_score = prediction.get(
                "score",
                0.0,
            )

            if not candidate_field:
                continue

            if candidate_field in excluded_fields:
                continue

            # Secondary candidates must still pass the
            # global semantic threshold.
            if (
                candidate_score
                < self.detector.threshold
            ):
                continue

            # Prevent very weak candidates from being
            # accepted only because their parser succeeds.
            if (
                candidate_score
                < minimum_relative_score
            ):
                continue

            metadata = self.field_metadata.get(
                candidate_field
            )

            if metadata is None:
                continue

            parser_strategy = (
                metadata.get(
                    "parser_strategy",
                    "",
                )
                .strip()
                .lower()
            )

            # A time parser confirms only that the clause
            # contains a valid time. It does not identify
            # which clinical time field owns that value.
            # Once a primary time field has been selected,
            # do not copy its value into other time fields.
            if (
                primary_parser_strategy == "time"
                and parser_strategy == "time"
            ):
                continue

            # A pure timing statement such as "the operation
            # finished at 10:25" must not also become a
            # narrative procedure description.
            if (
                primary_parser_strategy == "time"
                and candidate_field
                == "procedure_description"
            ):
                continue

            # One clause should not create duplicate event
            # records under overlapping event-list fields.
            #
            # For example, if unexpected_event is already
            # the primary field, the same statement should
            # not also be stored as complications.
            if (
                primary_parser_strategy == "event_list"
                and parser_strategy == "event_list"
            ):
                continue

            # Only parsers that provide sufficiently strict
            # field-specific confirmation may recover an
            # additional field.
            # Only parsers with strict confirmation may
            # recover an additional field.
            is_strict_strategy = (
                parser_strategy
                in self.strict_secondary_strategies
            )

            is_validated_measurement = (
                parser_strategy == "measurement"
                and candidate_field
                in self.strict_secondary_measurement_fields
            )

            if not (
                is_strict_strategy
                or is_validated_measurement
            ):
                continue

            validation_result = parse_and_validate(
                transcript=clause,
                metadata=metadata,
            )

            if not validation_result["accepted"]:
                continue

            # A generic word such as "no" can refer to a
            # different clinical concept. Therefore, a
            # negative Boolean cannot independently confirm
            # a secondary semantic candidate.
            if (
                parser_strategy == "boolean"
                and validation_result[
                    "converted_value"
                ] is False
                and candidate_field
                not in (
                    self.strict_secondary_negative_boolean_fields
                )
            ):
                continue
            result = self.create_extraction_result(
                clause=clause,
                field=candidate_field,
                candidate_field=candidate_field,
                semantic_score=candidate_score,
                metadata=metadata,
                validation_result=validation_result,
                matched_reference=prediction.get(
                    "matched_reference"
                ),
                selection_strategy=(
                    "secondary_parser_confirmed"
                ),
            )

            secondary_results.append(result)
            excluded_fields.add(candidate_field)

        return secondary_results
    def extract_group_structured_results(
        self,
        clause: str,
        primary_field: str,
        excluded_fields: set[str],
    ) -> list[dict[str, Any]]:
        group_results = []
        # Some schema groups contain tightly related fields
        # with different parser strategies. These companions
        # may be checked only when the accepted primary field
        # explicitly permits them.
        allowed_group_companions = {
            "blood_transfusion": {
                "blood_transfused_ml",
            },
            "blood_transfused_ml": {
                "blood_transfusion",
            },
            "esu_used": {
        "esu_serial_number",
            },
            "esu_serial_number": {
                "esu_used",
            },
            "tourniquet_used": {
                "tourniquet_pressure_mmhg",
                "tourniquet_duration_minutes",
                "tourniquet_serial_number",
            },
            "tourniquet_pressure_mmhg": {
                "tourniquet_used",
                "tourniquet_duration_minutes",
                "tourniquet_serial_number",
            },
            "tourniquet_duration_minutes": {
                "tourniquet_used",
                "tourniquet_pressure_mmhg",
                "tourniquet_serial_number",
            },
            "tourniquet_serial_number": {
                "tourniquet_used",
                "tourniquet_pressure_mmhg",
                "tourniquet_duration_minutes",
            },
            "irrigation_solution": {"irrigation_used"},
            "irrigation_used": {"irrigation_solution"},
            "laboratory_requests": {
                "specimen_collected",
            },
            "specimen_collected": {
                "laboratory_requests",
            },
        }

        # Find the schema path of the accepted primary field.
        primary_metadata = self.field_metadata.get(
            primary_field,
            {},
        )

        primary_schema_path = primary_metadata.get(
            "schema_path",
            "",
        ).strip()

        # A field without a schema path cannot define a group.
        if not primary_schema_path:
            return group_results

        # Example:
        # medications.prophylactic_antibiotic_given
        # becomes the group "medications".
        primary_group = primary_schema_path.split(
            ".",
            maxsplit=1,
        )[0]

        # Inspect every known metadata field.
        for candidate_field, metadata in (
            self.field_metadata.items()
        ):
            # Do not create a field that was already extracted.
            if candidate_field in excluded_fields:
                continue

            parser_strategy = metadata.get(
                "parser_strategy",
                "",
            ).strip().lower()
            is_allowed_companion = (
                candidate_field
                in allowed_group_companions.get(
                    primary_field,
                    set(),
                )
            )

            # Only strict structured parsers are allowed.
            # Generic Boolean, text, and event parsers are
            # deliberately excluded to reduce false positives.
            if (
                parser_strategy not in {
                    "structured",
                    "structured_list",
                }
                and not is_allowed_companion
            ):
                continue

            candidate_schema_path = metadata.get(
                "schema_path",
                "",
            ).strip()

            if not candidate_schema_path:
                continue

            candidate_group = (
                candidate_schema_path.split(
                    ".",
                    maxsplit=1,
                )[0]
            )

            # Only inspect fields from the same schema group.
            if candidate_group != primary_group:
                continue

            # The field is not accepted automatically.
            # Its dedicated parser must find a valid value.
            validation_result = parse_and_validate(
                transcript=clause,
                metadata=metadata,
            )

            if not validation_result["accepted"]:
                continue

            # This field was parser-confirmed rather than
            # semantically selected, so no semantic score or
            # matched reference is invented.
            result = self.create_extraction_result(
                clause=clause,
                field=candidate_field,
                candidate_field=candidate_field,
                semantic_score=0.0,
                metadata=metadata,
                validation_result=validation_result,
                matched_reference=None,
                selection_strategy=(
                    "group_parser_confirmed"
                ),
            )

            group_results.append(result)

            # Prevent this field from being added again.
            excluded_fields.add(candidate_field)

        return group_results

    def extract_related_event_results(
        self,
        clause: str,
        primary_field: str,
        excluded_fields: set[str],
    ) -> list[dict[str, Any]]:
        """
        Allow one clinical event to populate multiple
        event-list fields when each field-specific parser
        independently confirms the clause.
        """

        event_results = []

        primary_metadata = self.field_metadata.get(
            primary_field,
            {},
        )

        primary_parser_strategy = primary_metadata.get(
            "parser_strategy",
            "",
        ).strip().lower()

        # Only run multi-label event validation after an
        # event-list field has already been accepted.
        if primary_parser_strategy != "event_list":
            return event_results

        for candidate_field, metadata in (
            self.field_metadata.items()
        ):
            if candidate_field in excluded_fields:
                continue

            candidate_parser_strategy = metadata.get(
                "parser_strategy",
                "",
            ).strip().lower()

            # Only other event-list fields are eligible.
            if candidate_parser_strategy != "event_list":
                continue

            validation_result = parse_and_validate(
                transcript=clause,
                metadata=metadata,
            )

            # The candidate is accepted only when its own
            # field-specific parser confirms the event.
            if not validation_result["accepted"]:
                continue

            result = self.create_extraction_result(
                clause=clause,
                field=candidate_field,
                candidate_field=candidate_field,
                semantic_score=0.0,
                metadata=metadata,
                validation_result=validation_result,
                matched_reference=None,
                selection_strategy=(
                    "event_parser_confirmed"
                ),
            )

            event_results.append(result)
            excluded_fields.add(candidate_field)

        return event_results

    def find_below_threshold_event(
        self,
        clause: str,
        semantic_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Recover a structured clinical event when every
        BioLORD prediction is below the global threshold.

        The event candidate must:
        1. Have an absolute score of at least 0.20.
        2. Retain at least 70% of the best score.
        3. Use the event_list parser.
        4. Pass structured event validation.
        """

        top_score = semantic_result.get(
            "score",
            0.0,
        )

        for prediction in semantic_result.get(
            "top_predictions",
            [],
        ):
            candidate_field = prediction.get(
                "field"
            )

            candidate_score = prediction.get(
                "score",
                0.0,
            )

            if not candidate_field:
                continue

            if (
                candidate_score
                < self.event_rescue_minimum_score
            ):
                continue

            if (
                top_score > 0
                and candidate_score
                < (
                    top_score
                    * self.event_rescue_relative_ratio
                )
            ):
                continue

            metadata = self.field_metadata.get(
                candidate_field
            )

            if metadata is None:
                continue

            parser_strategy = (
                metadata.get(
                    "parser_strategy",
                    "",
                )
                .strip()
                .lower()
            )

            # Below-threshold recovery is restricted
            # to structured clinical events.
            if parser_strategy != "event_list":
                continue

            validation_result = (
                parse_and_validate(
                    transcript=clause,
                    metadata=metadata,
                )
            )

            if not validation_result[
                "accepted"
            ]:
                continue

            return {
                "field": candidate_field,
                "score": candidate_score,
                "metadata": metadata,
                "validation": validation_result,
                "matched_reference": (
                    prediction.get(
                        "matched_reference"
                    )
                ),
            }

        return None
    def extract_explicit_case_information_results(
        self,
        clause: str,
    ) -> list[dict[str, Any]]:
        """
        Recover multiple case-information fields when
        an explicit surgical-case label is present.

        Each field must still pass its dedicated parser.
        """

        normalized_clause = clause.lower()

        explicit_case_label = re.search(
            r"\b(?:"
            r"case\s*(?:id|number)|"
            r"(?:surgical|operative|operation)\s+case"
            r"(?:\s+(?:id|number))?"
            r")\b",
            normalized_clause,
        )

        if explicit_case_label is None:
            return []

        results = []

        for field_name in (
            "case_id",
            "date",
            "or_room_number",
        ):
            metadata = self.field_metadata.get(
                field_name
            )

            if metadata is None:
                continue

            validation_result = parse_and_validate(
                transcript=clause,
                metadata=metadata,
            )

            if not validation_result["accepted"]:
                continue

            results.append(
                self.create_extraction_result(
                    clause=clause,
                    field=field_name,
                    candidate_field=field_name,
                    semantic_score=0.0,
                    metadata=metadata,
                    validation_result=validation_result,
                    matched_reference=None,
                    selection_strategy=(
                        "explicit_case_parser_confirmed"
                    ),
                )
            )

        return results

    def extract_count_context_results(
        self,
        transcript: str,
        existing_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        accepted_fields = {
            result.get("field")
            for result in existing_results
            if result.get("status") == "accepted"
        }

        # Do not activate this contextual pass unless the
        # normal pipeline already confirmed a count problem.
        has_positive_count_discrepancy = any(
            result.get("status") == "accepted"
            and result.get("field") == "count_discrepancy"
            and result.get("value") is True
            for result in existing_results
        )

        # Contextual count recovery is needed only when
        # an actual discrepancy was confirmed. A valid
        # negative value must not activate this pass.
        if not has_positive_count_discrepancy:
            return []

        contextual_results = []

        related_fields = (
            "counts_correct",
            "count_actions",
            "unexpected_event",
        )

        for candidate_field in related_fields:
                # final_count_correct describes the overall final or
    # closing count, not an isolated sponge/instrument count.
            if (
            candidate_field == "counts_correct"
            and not re.search(
                r"\b(?:final|closing)\s+"
                r"(?:(?:sponge|instrument|needle|surgical)\s+)?"
                r"counts?\b"
                r".{0,40}\b"
                r"(?:correct|incorrect|not\s+correct|"
                r"did\s+not\s+reconcile|didn't\s+reconcile)\b",
                transcript,
                flags=re.IGNORECASE,
            )
        ):
                continue
            # Avoid inserting the same field twice.
            if candidate_field in accepted_fields:
                continue

            metadata = self.field_metadata.get(
                candidate_field
            )

            if metadata is None:
                continue

            validation_result = parse_and_validate(
                transcript=transcript,
                metadata=metadata,
            )

            # The complete transcript must contain valid
            # field-specific evidence.
            if not validation_result["accepted"]:
                continue

            contextual_result = (
                self.create_extraction_result(
                    clause=transcript,
                    field=candidate_field,
                    candidate_field=candidate_field,
                    semantic_score=0.0,
                    metadata=metadata,
                    validation_result=validation_result,
                    matched_reference=None,
                    selection_strategy=(
                        "context_parser_confirmed"
                    ),
                )
            )

            contextual_results.append(
                contextual_result
            )

            accepted_fields.add(
                candidate_field
            )

        return contextual_results

    def extract_explicit_labeled_results(self, clause):
        """Recover only fields explicitly identified by a clinical label.

        This extends the existing explicit-label path; it does not turn
        successful arbitrary parsers into unscored semantic predictions.
        """
        results = self._extract_legacy_explicit_labeled_results(clause)
        present = {r["field"] for r in results if r.get("status") == "accepted"}
        labels = {
            "preoperative_diagnosis": r"\bdiagnosis\s+(?:(?:established|documented|recorded|confirmed|identified)\s+)?before\s+(?:the\s+)?(?:surgery|operation|procedure)\s+(?:is|was)\b",
            "postoperative_diagnosis": r"\bdiagnosis\s+(?:(?:established|documented|recorded|confirmed|identified)\s+)?after\s+(?:the\s+)?(?:surgery|operation|procedure)\s+(?:is|was)\b",
            "known_allergies": r"\bpatient\s+(?:reports?|reported)\s+(?:a|an|documented|known)?\s*allerg(?:y|ies)\s+to\b",
                        "drain_inserted": (
                r"\b(?:a\s+)?"
                r"(?:\d+\s+(?:french|fr)\s+)?"
                r"jackson[-\s]?pratt\s+"
                r"(?:drain|train)\s+"
                r"(?:was\s+)?"
                r"(?:inserted|placed|positioned|"
                r"left\s+in\s+place)\b"
            ),
            "drain_details": (
                r"\b(?:a\s+)?"
                r"(?:\d+\s+(?:french|fr)\s+)?"
                r"jackson[-\s]?pratt\s+"
                r"(?:drain|train)\s+"
                r"(?:was\s+)?"
                r"(?:inserted|placed|positioned|"
                r"left\s+in\s+place)\b"
            ),
            "patient_name": r"\bas\s+(?:the\s+)?patient\s+name\b",
            "procedure_site": r"\b(?:surgical|procedure|operation)\s+site\s+(?:is|was)\b",
            "surgeon_1": r"\bprimary\s+surgeon\s+(?:for\s+(?:this|the)\s+(?:case|operation|procedure)\s+)?(?:is|was)\b",
            "anesthesiologist": r"\banesthesiologist\s*(?:is|was|,)\s*(?:dr\.?|doctor)\b",
            "incision_time": r"\b(?:skin\s+)?incision\s+(?:took\s+place|was\s+made|began|started)\s+at\b",
            "surgery_end_time": r"\b(?:operation|surgery|procedure)\s+(?:was\s+)?(?:completed|ended|finished|concluded)\s+at\b",
            "transfer_destination": r"\b(?:transferred|sent)\s+to\s+(?:the\s+)?(?:(?:surgical|medical|general|inpatient)\s+)?ward\b",
            "other_destination": (r"\b(?:before|after)\s+transfer\s+to\s+(?:the\s+)?"),
        }
        for field, label in labels.items():
            if field in present or not re.search(label, clause, re.I):
                continue
            metadata = self.field_metadata.get(field)
            if metadata is None:
                continue
            validation = parse_and_validate(clause, metadata)
            if not validation["accepted"]:
                continue
            results.append(self.create_extraction_result(
                clause=clause, field=field, candidate_field=None,
                semantic_score=None, metadata=metadata,
                validation_result=validation, matched_reference=None,
                selection_strategy="explicit_label_parser_confirmed"))
        return results

    def _extract_legacy_explicit_labeled_results(
        self,
        clause: str,
    ) -> list[dict[str, Any]]:
        """
        Extract fields from clauses containing explicit
        clinical labels that unambiguously identify their
        schema group.
        """

        normalized_clause = clause.lower()

        if re.search(
            r"\b(?:"
            r"(?:preoperative|pre-operative|pre-op)\s+"
            r"diagnosis|"
            r"diagnosis\s+before\s+(?:the\s+)?"
            r"(?:surgery|operation|procedure)"
            r")\b",
            normalized_clause,
        ):
            fields_to_check = (
                "preoperative_diagnosis",
            )

        elif re.search(
            r"\b(?:postoperative|post-operative|post-op)\s+"
            r"diagnosis\b",
            normalized_clause,
        ):
            fields_to_check = (
                "postoperative_diagnosis",
            )
        elif re.search(
            r"\b(?:surgical\s+)?"
            r"(?:procedure|operation)\s+"
            r"(?:was\s+)?completed\s+by\b",
            normalized_clause,
        ):
            fields_to_check = (
                "procedure_description",
            )

        elif re.search(
            r"\boperative\s+"
            r"(?:findings?|inspection|examination)\s+"
            r"(?:showed|revealed|demonstrated)\b",
            normalized_clause,
        ):
            fields_to_check = (
                "operative_findings",
            )
        elif re.search(
            r"\b(?:"
            r"(?:operative|patient)\s+position\s+"
            r"(?:is|was|:|-)|"
            r"(?:the\s+)?patient\s+"
            r"(?:was\s+)?positioned\s+(?:in|as)"
            r")\b",
            normalized_clause,
        ):
            standard_position_metadata = (
                self.field_metadata.get(
                    "patient_position"
                )
            )

            standard_position_validation = None

            if standard_position_metadata is not None:
                standard_position_validation = (
                    parse_and_validate(
                        transcript=clause,
                        metadata=(
                            standard_position_metadata
                        ),
                    )
                )

            if (
                standard_position_validation is not None
                and standard_position_validation[
                    "accepted"
                ]
            ):
                fields_to_check = (
                    "patient_position",
                )
            else:
                fields_to_check = (
                    "other_position",
                )

        elif re.search(
            r"\b(?:"
            r"planned\s+(?:procedure|operation)|"
            r"procedure\s+(?:performed\s+)?(?:is|was)|"
            r"(?:dr\.?|doctor)\s+[a-zà-öø-ÿ'’-]+"
            r"(?:\s+[a-zà-öø-ÿ'’-]+){1,3}\s+"
            r"performed\s+(?:an?|the)|"
            r"(?:the\s+)?surgeon\s+performed\s+(?:an?|the)"
            r")\b",
            normalized_clause,
        ):
            fields_to_check = (
                "surgeon_1",
                "procedure_name",
                "procedure_site",
                "laterality",
            )

        elif re.search(
            r"\b(?:segment|portion|piece|sample)\s+of\s+"
            r".{1,80}\bcollected\s+as\s+"
            r"(?:a\s+)?(?:specimen|sample)\b",
            normalized_clause,
        ):
            fields_to_check = (
                "specimen_collected",
                "specimen_details",
                "laboratory_requests",
            )

        elif re.search(
            r"\b(?:specimen|sample)\s+"
            r"(?:was\s+)?(?:sent|transferred|submitted)\s+to\s+"
            r".{1,80}\b(?:laboratory|lab)\b",
            normalized_clause,
        ):
            fields_to_check = (
                "specimen_sent_to_external_lab",
                "external_lab_name",
            )

        elif re.search(
            r"\b(?:container\s+(?:was\s+)?marked\s+urgent|"
            r"transported\s+on\s+ice)\b",
            normalized_clause,
        ):
            fields_to_check = (
                "specimen_other_notes",
            )
        elif re.search(
            r"\b(?:"
            r"(?:the\s+)?patient\s+(?:remained|was)\s+"
            r"intubated|"
            r"(?:the\s+)?patient\b.{0,80}?\b"
            r"(?:was\s+)?transferred\s+to|"
            r"(?:was\s+)?transferred\s+to"
            r")\b",
            normalized_clause,
        ):
            fields_to_check = (
                "outcome_intubated",
                "other_destination",
            )

        elif re.search(
            r"\b(?:"
            r"responsible\s+physician|"
            r"recovery\s+nurse|"
            r"receiving\s+floor\s+nurse|"
            r"floor\s+nurse"
            r")\b",
            normalized_clause,
        ):
            fields_to_check = (
                "physician_name",
                "recovery_nurse_name",
                "floor_nurse_name",
            )

        else:
            return []

        results = []

        for field_name in fields_to_check:
            metadata = self.field_metadata.get(
                field_name
            )

            if metadata is None:
                continue

            validation_result = parse_and_validate(
                transcript=clause,
                metadata=metadata,
            )

            if not validation_result["accepted"]:
                continue

            results.append(
                self.create_extraction_result(
                    clause=clause,
                    field=field_name,
                    candidate_field=field_name,
                    semantic_score=0.0,
                    metadata=metadata,
                    validation_result=validation_result,
                    matched_reference=None,
                    selection_strategy=(
                        "explicit_label_parser_confirmed"
                    ),
                )
            )

        return results

    def process_transcript(
        self,
        transcript: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Segment a transcript, classify each clause,
        parse its values, and validate the results.
        """

        clauses = self.segmenter.segment(
            transcript
        )

        final_results = []

        for clause in clauses:
            # At least ten candidates are required for
            # fallback and multi-field extraction.
            semantic_result = (
                self.detector.predict(
                    clause,
                    top_k=max(top_k, 10),
                )
            )

            predicted_field = semantic_result[
                "predicted_field"
            ]

            candidate_field = semantic_result[
                "candidate_field"
            ]

            if predicted_field is None:
                event_rescue = (
                    self.find_below_threshold_event(
                        clause=clause,
                        semantic_result=semantic_result,
                    )
                )

                explicit_labeled_results = []

                # Explicit-label recovery runs only when
                # the specialized event rescue did not
                # already validate the clause.
                if event_rescue is None:
                    explicit_labeled_results = (
                        self.extract_explicit_labeled_results(
                            clause=clause,
                        )
                    )

                if event_rescue is not None:
                    rescue_result = (
                        self.create_extraction_result(
                            clause=clause,
                            field=event_rescue[
                                "field"
                            ],
                            candidate_field=(
                                candidate_field
                            ),
                            semantic_score=(
                                event_rescue["score"]
                            ),
                            metadata=event_rescue[
                                "metadata"
                            ],
                            validation_result=(
                                event_rescue[
                                    "validation"
                                ]
                            ),
                            matched_reference=(
                                event_rescue[
                                    "matched_reference"
                                ]
                            ),
                            selection_strategy=(
                                "parser_guided_event_rescue"
                            ),
                        )
                    )

                    final_results.append(
                        rescue_result
                    )

                elif explicit_labeled_results:
                    final_results.extend(
                        explicit_labeled_results
                    )

                else:
                    final_results.append(
                        {
                            "clause": clause,
                            "status": (
                                "semantic_rejected"
                            ),
                            "field": None,
                            "candidate_field": (
                                candidate_field
                            ),
                            "semantic_score": (
                                semantic_result[
                                    "score"
                                ]
                            ),
                            "value": None,
                            "reason": (
                                "The semantic score was "
                                "below the threshold."
                            ),
                            "selection_strategy": (
                                "semantic_rejected"
                            ),
                        }
                    )

                continue

            metadata = self.field_metadata.get(
                predicted_field
            )

            if metadata is None:
                final_results.append(
                    {
                        "clause": clause,
                        "status": (
                            "metadata_missing"
                        ),
                        "field": predicted_field,
                        "candidate_field": (
                            candidate_field
                        ),
                        "semantic_score": (
                            semantic_result[
                                "score"
                            ]
                        ),
                        "value": None,
                        "reason": (
                            "No metadata row was found "
                            "for the predicted field."
                        ),
                        "selection_strategy": (
                            "primary_semantic"
                        ),
                    }
                )

                continue

            # Validate BioLORD's primary prediction.
            validation_result = (
                parse_and_validate(
                    transcript=clause,
                    metadata=metadata,
                )
            )

            selected_field = predicted_field
            selected_metadata = metadata

            selected_score = semantic_result[
                "score"
            ]

            selected_reference = (
                semantic_result.get(
                    "matched_reference"
                )
            )

            selection_strategy = (
                "primary_semantic"
            )

            original_predicted_field = (
                predicted_field
            )

            # If the primary parser rejects the value,
            # first try alternatives that passed the normal
            # semantic threshold. If none succeeds, allow the
            # constrained event rescue to inspect lower-ranked
            # event candidates.
            if not validation_result["accepted"]:
                event_rescue_used = False

                fallback_result = (
                    self.find_parser_guided_fallback(
                        clause=clause,
                        semantic_result=(
                            semantic_result
                        ),
                        rejected_field=(
                            predicted_field
                        ),
                    )
                )

                if fallback_result is None:
                    fallback_result = (
                        self.find_below_threshold_event(
                            clause=clause,
                            semantic_result=(
                                semantic_result
                            ),
                        )
                    )

                    event_rescue_used = (
                        fallback_result is not None
                    )

                if fallback_result is not None:
                    selected_field = (
                        fallback_result["field"]
                    )

                    selected_metadata = (
                        fallback_result["metadata"]
                    )

                    selected_score = (
                        fallback_result["score"]
                    )

                    selected_reference = (
                        fallback_result[
                            "matched_reference"
                        ]
                    )

                    validation_result = (
                        fallback_result[
                            "validation"
                        ]
                    )

                    if event_rescue_used:
                        selection_strategy = (
                            "parser_guided_event_rescue"
                        )
                    else:
                        selection_strategy = (
                            "parser_guided_fallback"
                        )

            primary_result = (
                self.create_extraction_result(
                    clause=clause,
                    field=selected_field,
                    candidate_field=(
                        candidate_field
                    ),
                    semantic_score=selected_score,
                    metadata=selected_metadata,
                    validation_result=(
                        validation_result
                    ),
                    matched_reference=(
                        selected_reference
                    ),
                    selection_strategy=(
                        selection_strategy
                    ),
                )
            )

            final_results.append(primary_result)

            # Search for additional facts only after
            # the selected field has been validated.
                        # If the semantic candidate was rejected,
            # attempt strict explicit-label recovery.
            if primary_result[
                "status"
            ] != "accepted":
                explicit_labeled_results = (
                    self.extract_explicit_labeled_results(
                        clause=clause,
                    )
                )

                if explicit_labeled_results:
                    # Remove the rejected primary result
                    # and replace it with validated fields.
                    final_results.pop()

                    final_results.extend(
                        explicit_labeled_results
                    )

                continue

            excluded_fields = {
                primary_result["field"],
                selected_field,
                original_predicted_field,
            }

            # An accepted primary field may share its
            # clause with other strictly labeled fields.
            explicit_labeled_results = (
                self.extract_explicit_labeled_results(
                    clause=clause,
                )
            )

            for explicit_result in (
                explicit_labeled_results
            ):
                explicit_result_field = (
                    explicit_result["field"]
                )

                if (
                    explicit_result_field
                    in excluded_fields
                ):
                    continue

                final_results.append(
                    explicit_result
                )

                excluded_fields.add(
                    explicit_result_field
                )

            secondary_results = (
                self.extract_secondary_results(
                    clause=clause,
                    semantic_result=semantic_result,
                    excluded_fields=(
                        excluded_fields
                    ),
                )
            )

            final_results.extend(
                secondary_results
            )
            group_results = (
                self.extract_group_structured_results(
                    clause=clause,
                    primary_field=primary_result["field"],
                    excluded_fields=(
                        excluded_fields
                    ),
                )
            )

            final_results.extend(
                group_results
            )
            # Allow one validated clinical event to populate
            # another event-list field when that field's own
            # strict parser independently confirms the clause.
            event_results = (
                self.extract_related_event_results(
                    clause=clause,
                    primary_field=(
                        primary_result["field"]
                    ),
                    excluded_fields=(
                        excluded_fields
                    ),
                )
            )

            final_results.extend(
                event_results
            )
            case_information_results = (
                self.extract_explicit_case_information_results(
                    clause=clause,
                )
            )

            for case_result in (
                case_information_results
            ):
                case_field = case_result["field"]

                if case_field in excluded_fields:
                    continue

                final_results.append(
                    case_result
                )

                excluded_fields.add(
                    case_field
                )
        count_context_results = (
            self.extract_count_context_results(
                transcript=transcript,
                existing_results=final_results,
            )
        )

        final_results.extend(
            count_context_results
        )

        return final_results


def print_pipeline_results(
    transcript: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Print the final integrated extraction results.
    """

    print("\n" + "=" * 70)
    print("TRANSCRIPT")
    print(transcript)

    print("\nFINAL EXTRACTION RESULTS")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {index}:")
        print("  Text:", result["clause"])
        print("  Status:", result["status"])
        print("  Field:", result["field"])
        print(
            "  Semantic score:",
            result["semantic_score"],
        )

        print(
            "  Selection strategy:",
            result.get(
                "selection_strategy"
            ),
        )

        if result["status"] in {
            "accepted",
            "value_rejected",
        }:
            print(
                "  Parser:",
                result["parser_strategy"],
            )

            print(
                "  Schema path:",
                result["schema_path"],
            )

            print(
                "  Extracted value:",
                result["extracted_value"],
            )

        print(
            "  Final value:",
            result["value"],
        )

        print(
            "  Reason:",
            result["reason"],
        )


if __name__ == "__main__":
    pipeline = ClinicalExtractionPipelineV1()

    test_transcripts = [
        (
            "Unexpected arterial bleeding occurred "
            "and was controlled with clips."
        ),
        (
            "The final counts were correct, "
            "the procedure was completed successfully, "
            "and the patient remained stable."
        ),
        (
            "The operative findings showed an inflamed "
            "appendix without perforation and no "
            "intraoperative complications occurred."
        ),
    ]

    for test_transcript in test_transcripts:
        results = pipeline.process_transcript(
            transcript=test_transcript,
            top_k=10,
        )

        print_pipeline_results(
            transcript=test_transcript,
            results=results,
        )
