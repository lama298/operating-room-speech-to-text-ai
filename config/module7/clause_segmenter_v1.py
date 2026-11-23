import re


class ClauseSegmenterV1:
    """
    Separates a transcript containing multiple clinical facts
    into smaller standalone clauses for the single-label
    semantic detector.
    """

    CLINICAL_CUES = [
        r"\b(?:surgical|procedure|operation) site\b",
        r"\blaterality\b",
        r"\bdiagnosis\b",
        r"\bprocedure\b",
        r"\banesthesia\b",
        r"\banastasia\b",
        r"\bpositioned\b",
        r"\bpatient position\b",
        r"\badministered\b",
        r"\bmedication\b",
        r"\bantibiotic\b",
        r"\bcefazolin\b",
        r"\bcifazolin\b",
        r"\bblood loss\b",
        r"\bcounts?\b",
        r"\bcount discrepancy\b",
        r"\btransferred\b",
        r"\bdestination\b",
        r"\bstable\b",
        r"\bfair\b",
        r"\bpatient status\b",
        r"\bspecimen\b",
        r"\bpathology\b",
        r"\boperative findings\b",
        r"\bcomplications?\b",
        r"\bunexpected\b",
        r"\bbleeding\b",
        r"\bblood pressure\b",
        r"\bconverted\b",
        r"\bsurgical field\b",
        r"\b(?:extubated|intubated)\b",
        r"\b(?:irrigation|irrigated|washed|rinsed|flushed)\b",
        r"\b(?:lactated\s+ringer'?s?|ringer'?s\s+solution|normal\s+saline)\b",
        r"\b(?:total\s+)?(?:intravenous|iv)\s+fluid(?:s)?\b",
        r"\bfluid\s+volume\b",
        r"\bjackson[-\s]?pratt\s+(?:drain|train)\b",
        r"\b(?:implant|mesh|prosthesis|plate|screw)\b",
    ]

    def normalize(self, transcript):
        """
        Clean basic spacing without changing clinical wording.
        """

        transcript = transcript.strip()

        transcript = re.sub(
            r"\s+",
            " ",
            transcript,
        )

        transcript = re.sub(
            r"\s+([,.!?])",
            r"\1",
            transcript,
        )

        return transcript

    def contains_clinical_cue(self, text):
        """
        Return True when a fragment contains a clinical cue.
        """

        lowered_text = text.lower()

        return any(
            re.search(pattern, lowered_text)
            for pattern in self.CLINICAL_CUES
        )

    def split_sentences(self, transcript):
        """
        Split the transcript at strong punctuation while
        preserving periods inside common professional titles.
        """

        protected_transcript = re.sub(
            r"\b(Dr|Mr|Mrs|Ms|Prof)\.",
            r"\1<PERIOD>",
            transcript,
            flags=re.IGNORECASE,
        )

        parts = re.split(
            r"(?<=[.!?])\s+",
            protected_transcript,
        )

        return [
            part.replace("<PERIOD>", ".").strip()
            for part in parts
            if part.strip()
        ]

    def split_special_patterns(self, clause):
        """
        Handle clinical constructions without a normal
        conjunction boundary.
        """

        patterns = [
            # "appendectomy under general anesthesia"
            (
                r"\s+(?=under\s+"
                r"(?:general|regional|local|spinal)"
                r"\s+anesthesia\b)"
            ),

            # "sent to pathology with estimated blood loss"
            (
                r"\s+(?=with\s+estimated\s+"
                r"blood\s+loss\b)"
            ),

            # "transferred to PACU in stable condition"
            (
                r"\s+(?=in\s+(?:stable|fair)"
                r"\s+condition\b)"
            ),
        ]
                # Split a surgical case statement into its date and OR room facts.
        if re.search(
            r"\b(?:case|surgery|operation|procedure)\b"
            r".{0,80}\b(?:was\s+)?(?:performed|took\s+place)\b",
            clause,
            flags=re.IGNORECASE,
        ):
            patterns.extend([
                r"\s+(?=on\s+[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b)",
                r"\s+(?=in\s+(?:operating\s+room|OR\s+room)\s+"
                r"[A-Za-z0-9]+\b)",
            ])

        parts = [clause]

        for pattern in patterns:
            updated_parts = []

            for part in parts:
                split_parts = re.split(
                    pattern,
                    part,
                    flags=re.IGNORECASE,
                )

                updated_parts.extend(
                    split_part.strip(" ,")
                    for split_part in split_parts
                    if split_part.strip(" ,")
                )

            parts = updated_parts

        return parts

    def is_event_response_pair(
        self,
        left: str,
        right: str,
    ) -> bool:
        """
        Return True when the right side describes the
        response, intervention, or follow-up to an event
        stated on the left side.
        """

        event_pattern = (
            r"\b(?:unexpected|unplanned|sudden|adverse|"
            r"complication|bleeding|blood pressure|"
            r"discrepancy|malfunction|converted|"
            r"decrease|increase)\b"
        )

        response_pattern = (
            r"\b(?:"
            r"(?:the\s+)?(?:anesthesia|surgical|clinical|medical)"
            r"\s+team\s+was\s+(?:notified|informed|called)|"
            r"(?:was|were)\s+(?:controlled|treated|managed|"
            r"corrected|checked|repaired|replaced|resolved)|"
            r"^\s*(?:controlled|treated|managed|corrected|"
            r"checked|repaired|replaced|resolved)\b|"
            r"(?:corrective|additional)\s+action\s+was\s+taken"
            r")\b"
        )

        left_contains_event = re.search(
            event_pattern,
            left,
            flags=re.IGNORECASE,
        )

        right_contains_response = re.search(
            response_pattern,
            right,
            flags=re.IGNORECASE,
        )

        return bool(
            left_contains_event
            and right_contains_response
        )

    def find_connector_split(self, clause):
        """
        Find a conjunction or comma separating two clinical
        facts.
        """

        connector_pattern = re.compile(
            r",\s+(?:(?:and|while)\s+)?|"
            r"\s+(?:and|while)\s+",
            flags=re.IGNORECASE,
        )

        for match in connector_pattern.finditer(clause):
            left = clause[:match.start()].strip(" ,")
            right = clause[match.end():].strip(" ,")

            if not left or not right:
                continue

            left_is_clinical = (
                self.contains_clinical_cue(left)
            )

            right_is_clinical = (
                self.contains_clinical_cue(right)
            )

            if left_is_clinical and right_is_clinical:
    # Preserve an event and its associated response
    # as one complete clinical statement.
                if self.is_event_response_pair(
                    left,
                    right,
                ):
                    continue

                return left, right

        return None

    def split_connectors_recursively(self, clause):
        """
        Continue splitting until no valid clinical connector
        remains.
        """

        split_result = self.find_connector_split(
            clause
        )

        if split_result is None:
            return [clause]

        left, right = split_result

        left_parts = self.split_connectors_recursively(
            left
        )

        right_parts = self.split_connectors_recursively(
            right
        )

        return left_parts + right_parts

    def restore_punctuation(self, clause):
        """
        Give each returned clause terminal punctuation.
        """

        clause = clause.strip(" ,")

        if clause and clause[-1] not in ".!?":
            clause += "."

        return clause

    def restore_clause_context(self, clauses):
        """
        Convert incomplete fragments into complete standalone
        clinical statements.
        """

        restored_clauses = []

        for clause in clauses:
            cleaned_clause = (
                clause.strip().rstrip(".?!")
            )

            lowercase_clause = cleaned_clause.lower()

            # Restore anesthesia fragments.
            if lowercase_clause.startswith(
                "under general anesthesia"
            ):
                cleaned_clause = (
                    "General anesthesia was administered"
                )

            elif lowercase_clause.startswith(
                "under regional anesthesia"
            ):
                cleaned_clause = (
                    "Regional anesthesia was administered"
                )

            elif lowercase_clause.startswith(
                "under local anesthesia"
            ):
                cleaned_clause = (
                    "Local anesthesia was administered"
                )

            elif lowercase_clause.startswith(
                "under spinal anesthesia"
            ):
                cleaned_clause = (
                    "Spinal anesthesia was administered"
                )

            # Restore pathology fragments.
            elif lowercase_clause.startswith(
                "sent to pathology"
            ):
                specimen_subject = "The specimen"

                if restored_clauses:
                    previous_clause = (
                        restored_clauses[-1].lower()
                    )

                    if (
                        "appendix specimen"
                        in previous_clause
                    ):
                        specimen_subject = (
                            "The appendix specimen"
                        )

                cleaned_clause = (
                    f"{specimen_subject} "
                    "was sent to pathology"
                )

            # Restore estimated-blood-loss fragments.
            elif lowercase_clause.startswith(
                "with estimated blood loss of "
            ):
                prefix = (
                    "with estimated blood loss of "
                )

                blood_loss_value = cleaned_clause[
                    len(prefix):
                ]

                cleaned_clause = (
                    "Estimated blood loss was "
                    f"{blood_loss_value}"
                )

            # Restore patient-condition fragments.
            elif re.fullmatch(
                r"in\s+(?:stable|fair)\s+condition",
                lowercase_clause,
            ):
                condition = lowercase_clause.split()[1]

                cleaned_clause = (
                    f"The patient was in "
                    f"{condition} condition"
                )

            if cleaned_clause:
                cleaned_clause = (
                    cleaned_clause[0].upper()
                    + cleaned_clause[1:]
                )

                restored_clauses.append(
                    cleaned_clause + "."
                )

        return restored_clauses

    def segment(self, transcript):
        """
        Run the complete segmentation pipeline.

        Returns:
            list[str]: Complete standalone clinical clauses.
        """

        normalized = self.normalize(transcript)

        if not normalized:
            return []

        sentences = self.split_sentences(
            normalized
        )

        final_clauses = []

        for sentence in sentences:
            special_parts = (
                self.split_special_patterns(
                    sentence
                )
            )

            for part in special_parts:
                connector_parts = (
                    self.split_connectors_recursively(
                        part
                    )
                )

                final_clauses.extend(
                    self.restore_punctuation(clause)
                    for clause in connector_parts
                    if clause.strip()
                )

        return self.restore_clause_context(
            final_clauses
        )


if __name__ == "__main__":
    segmenter = ClauseSegmenterV1()

    test_transcripts = [
        (
            "The preoperative diagnosis is acute appendicitis "
            "and the procedure is laparoscopic appendectomy "
            "under general anesthesia."
        ),
        (
            "The patient was positioned supine and one gram "
            "of cefazolin was administered before incision."
        ),
        (
            "The appendix specimen was collected and sent "
            "to pathology with estimated blood loss of "
            "50 milliliters."
        ),
        (
            "The final counts were correct, the procedure "
            "was completed successfully, and the patient "
            "remained stable."
        ),
        (
            "The final instrument, sponge, and needle "
            "counts were correct."
        ),
        (
            "The patient was transferred to the "
            "post-anesthesia care unit in stable condition."
        ),
    ]

    for index, transcript in enumerate(
        test_transcripts,
        start=1,
    ):
        print(f"\nTEST {index}")
        print("Input:", transcript)
        print("Clauses:")

        clauses = segmenter.segment(
            transcript
        )

        for clause_number, clause in enumerate(
            clauses,
            start=1,
        ):
            print(
                f"  {clause_number}. {clause}"
            )