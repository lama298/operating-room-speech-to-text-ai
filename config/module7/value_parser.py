import re
from typing import Any
from datetime import datetime


def extract_value(
    transcript: str,
    matched_keyword: str,
) -> str | None:
    """
    Extracts the spoken value that appears after a matched keyword.

    This function belongs to the original keyword-based baseline.
    """
    normalized_transcript = transcript.strip()
    keyword_pattern = re.escape(matched_keyword)

    match = re.search(
        keyword_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    # Select everything that appears after the matched keyword.
    value = normalized_transcript[match.end():].strip()

    # Remove common connecting words and punctuation.
    value = re.sub(
        r"^[\s:,\-]*(?:(?:is|was|are|equal to|at|as)\b)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.strip(" .,:;-")

    if not value:
        return None

    return value


# Accepted spoken forms for each configured unit.
UNIT_PATTERNS = {
    "ml": r"(?:ml|milliliters?|cc)",
    "celsius": (
        r"(?:degrees?\s+celsius|degrees?\s+c|"
        r"°\s*c|celsius|C)"
    ),
        "mmhg": (
        r"(?:mm\s*hg|mm\s+of\s+mercury|"
        r"millimeters?\s+of\s+mercury)"
    ),
    "minutes": (
        r"(?:minutes?|mins?|min)"
    ),
}
CATEGORY_PATTERNS = {
    "general anesthesia": (
        r"\bgeneral(?:\s+anesthesia)?\b"
    ),
    "regional anesthesia": (
        r"\bregional(?:\s+anesthesia)?\b"
    ),
    "local anesthesia": (
        r"\blocal(?:\s+anesthesia)?\b"
    ),
    "spinal anesthesia": (
        r"\bspinal(?:\s+anesthesia)?\b"
    ),
    "epidural anesthesia": (
        r"\bepidural(?:\s+anesthesia)?\b"
    ),
    "monitored anesthesia care": (
        r"\b(?:monitored anesthesia care|mac)\b"
    ),
    "pacu": (
        r"\b(?:pacu|post[-\s]+anesthesia care unit|"
        r"post[-\s]+anaesthesia care unit)\b"
    ),
     "icu":r"\b(?:icu|intensive\s+care\s+unit)\b",
     "clean_contaminated": (
        r"\b(?:clean[-\s_]+contaminated|"
        r"class\s*(?:ii|2))\b"
    ),
}


def extract_measurement(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extracts a numeric measurement with the unit
    specified in the field metadata.
    """
    normalized_transcript = transcript.strip()
    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    field_evidence_patterns = {
        "estimated_blood_loss_ml": (
            r"\b(?:estimated\s+)?blood\s+loss\b|\bebl\b"
        ),
        "total_fluids_ml": (
            r"\btotal\s+(?:intravenous\s+|iv\s+)?"
            r"fluid\s+volume\b|"
            r"\btotal\s+fluids?\b|"
            r"\b(?:received|given|administered)\b"
            r".{0,50}\bfluids?\b|"
            r"\bfluids?\b.{0,30}"
            r"\b(?:received|given|administered)\b"
        ),
        "blood_transfused_ml": (
            r"\b(?:blood\s+)?transfusion\b|"
            r"\bpacked\s+(?:red\s+)?(?:blood\s+)?cells\b|"
            r"\btransfused\b"
        ),
    }

    evidence_pattern = field_evidence_patterns.get(
        field_name
    )

    if (
        evidence_pattern is not None
        and not re.search(
            evidence_pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )
    ):
        return None

    unit = metadata.get("unit", "").strip().lower()
    unit_pattern = UNIT_PATTERNS.get(unit)

    if unit_pattern:
        pattern = (
            r"(?<![\w,])"
            r"(-?(?:\d{1,3}(?:,\d{3})+|"
            r"\d+)(?:\.\d+)?)\s*"
            rf"(?:{unit_pattern})(?=\s|[.,;:]|$)"
        )
    else:
        pattern = pattern = (
            r"(?<![\w,])"
            r"(-?(?:\d{1,3}(?:,\d{3})+|"
            r"\d+)(?:\.\d+)?)"
            r"(?=\s|[.,;:]|$)"
        )

    match = re.search(
        pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).replace(",", "")


def extract_boolean(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extracts Boolean evidence and normalizes it to
    either "yes" or "no".

    Some Boolean fields require field-specific clinical
    wording instead of the literal words yes and no.
    """

    normalized_transcript = transcript.strip().lower()

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()
    field_patterns = {
        "specimen_collected": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:specimen|sample)\s+(?:was\s+)?collected|"
                r"(?:specimen|sample)\s+(?:was\s+)?not\s+collected"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:specimen|sample)\b.{0,30}\b"
                r"(?:collected|obtained|retrieved|excised)|"
                r"(?:collected|obtained|retrieved|excised)"
                r"\b.{0,30}\bas\s+(?:a\s+)?"
                r"(?:specimen|sample)"
                r")\b"
            ),
        },

        "specimen_sent_to_external_lab": {
            "negative": (
                r"\b(?:specimen|sample)\b"
                r".{0,50}\b(?:was\s+)?not\s+"
                r"(?:sent|transferred)\s+to\s+"
                r"(?:"
                r"(?:an?\s+)?"
                r"(?:external|outside|off[-\s]?site)\s+"
                r"(?:lab|laboratory)|"
                r"(?:the\s+)?"
                r"(?:[A-Za-z-]+\s+){1,4}"
                r"(?:Lab|Laboratory)"
                r")\b"
            ),
            "positive": (
                r"\b(?:specimen|sample)\b"
                r".{0,50}\b"
                r"(?:sent|transferred)\s+to\s+"
                r"(?:"
                r"(?:an?\s+)?"
                r"(?:external|outside|off[-\s]?site)\s+"
                r"(?:lab|laboratory)|"
                r"(?:the\s+)?"
                r"(?:[A-Za-z-]+\s+){1,4}"
                r"(?:Lab|Laboratory)"
                r")\b"
            ),
        },

        "antibiotic_prophylaxis_given": {
            "negative": (
                r"\b(?:no\s+prophylactic\s+antibiotic|"
                r"antibiotic\s+prophylaxis\s+was\s+not\s+"
                r"(?:administered|given)|"
                r"prophylactic\s+antibiotic\s+was\s+not\s+"
                r"(?:administered|given))\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:antibiotic(?:s)?\s+prophylaxis|"
                r"prophylactic\s+antibiotic(?:s)?)"
                r".{0,50}\b"
                r"(?:administered|given|provided)\b"
                r"|"
                r"(?:administered|given|provided)"
                r".{0,40}\b"
                r"(?:for\s+)?antibiotic(?:s)?\s+prophylaxis\b"
                r"|"
                r"(?:administered|given)"
                r".{0,30}\bbefore\s+(?:the\s+)?incision\b"
                r")"
            ),
        },
        "drain_inserted": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:[a-z0-9][a-z0-9-]*\s+){0,5}"
                r"(?:drain|bleak\s+train)\s+"
                r"(?:was\s+)?(?:inserted|placed)|"
                r"(?:[a-z0-9][a-z0-9-]*\s+){0,5}"
                r"(?:drain|bleak\s+train)\s+"
                r"(?:was\s+)?not\s+(?:inserted|placed)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:[a-z0-9][a-z0-9-]*\s+){0,5}drain|"
                r"(?:[a-z0-9][a-z0-9-]*\s+){0,5}"
                r"(?:bleak|jackson\s+pratt)\s+train"
                r")\s+"
                r"(?:was\s+)?"
                r"(?:inserted|placed|positioned|"
                r"left\s+in\s+place)\b"
            ),
        },
                "implant_used": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:implant|prosthesis|plate|mesh|"
                r"screw|device)\s+(?:was\s+)?"
                r"(?:used|implanted|inserted|placed)|"
                r"(?:implant|prosthesis|plate|mesh|"
                r"screw|device)\s+(?:was\s+)?not\s+"
                r"(?:used|implanted|inserted|placed)"
                r")\b"
            ),
            "positive": (
                r"\b(?:implant|prosthesis|plate|mesh|"
                r"screw|device)\b"
                r".{0,60}\b"
                r"(?:used|implanted|inserted|placed)\b"
            ),
        },

        "counts_correct": {
            "negative": (
                r"\b(?:counts?|final\s+counts?)\b"
                r".{0,30}\b(?:incorrect|not\s+correct|"
                r"did\s+not\s+reconcile|"
                r"didn't\s+reconcile|"
                r"(?<!no\s)discrepancy\s+"
                r"(?:was\s+)?identified)\b"
            ),
            "positive": (
                r"\b(?:counts?|final\s+counts?)\b"
                r".{0,50}\b(?:correct|matched|reconciled)\b"
            ),
        },

        "count_discrepancy": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:count\s+)?discrepancy|"
                r"discrepancy\s+was\s+not\s+identified"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"count\s+discrepancy|"
                r"discrepancy\s+(?:was\s+)?(?:identified|found|detected)"
                r"(?:\s+(?:during|in)\s+(?:the\s+)?"
                r"(?:final\s+)?(?:surgical\s+)?count)|"
                r"count\s+mismatch|"
                r"incorrect\s+count|"
                r"counts?\s+(?:was|were)\s+incorrect|"
                r"(?:count|counts)\s+(?:did\s+not|didn't)\s+"
                r"reconcile|"
                r"missing\s+(?:counted\s+)?item"
                r")\b"
            ),
        },
                "tourniquet_used": {
            "negative": (
                r"\b(?:"
                r"no\s+tourniquet\s+(?:was\s+)?"
                r"(?:used|applied|inflated)|"
                r"tourniquet\s+(?:was\s+)?not\s+"
                r"(?:used|applied|inflated)"
                r")\b"
            ),
            "positive": (
                r"\btourniquet\s+(?:was\s+)?"
                r"(?:used|applied|inflated)\b"
            ),
        },
                "outcome_extubated": {
            "negative": (
                r"\b(?:not\s+extubated|"
                r"extubation\s+(?:was\s+)?not\s+performed)\b"
            ),
            "positive": (
                r"\b(?:extubated|"
                r"extubation\s+(?:was\s+)?performed)\b"
            ),
        },
                "allergy_status_confirmed": {
            "negative": (
                r"\b(?:"
                r"allerg(?:y|ies)(?:\s+status)?\s+"
                r"(?:verification|review)\s+"
                r"(?:was\s+)?"
                r"(?:not\s+completed|pending|incomplete)|"
                r"allerg(?:y|ies)(?:\s+status)?\s+"
                r"(?:was|were)\s+not\s+"
                r"(?:verified|confirmed|reviewed)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"allerg(?:y|ies)(?:\s+status)?\s+"
                r"(?:verification|review)\s+"
                r"(?:was\s+)?(?:completed|confirmed)|"
                r"allerg(?:y|ies)(?:\s+status)?\s+"
                r"(?:was|were)\s+"
                r"(?:verified|confirmed|reviewed)"
                r")\b"
            ),
        },
                "isolation_required": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:contact\s+|airborne\s+|"
                r"droplet\s+|protective\s+)?"
                r"isolation\s+(?:is\s+)?required|"
                r"(?:contact\s+|airborne\s+|"
                r"droplet\s+|protective\s+)?"
                r"isolation\s+(?:is\s+)?"
                r"(?:not\s+required|not\s+indicated|"
                r"unnecessary)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:contact\s+|airborne\s+|"
                r"droplet\s+|protective\s+)?"
                r"isolation\s+(?:is\s+)?"
                r"(?:required|indicated|needed|ordered)|"
                r"(?:the\s+)?patient\s+requires\s+"
                r"(?:contact\s+|airborne\s+|"
                r"droplet\s+|protective\s+)?isolation"
                r")\b"
            ),
        },
        "fall_risk": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:identified\s+)?fall\s+risk|"
                r"(?:the\s+)?patient\s+is\s+not\s+"
                r"at\s+risk\s+of\s+falling|"
                r"fall\s+risk\s+(?:is\s+)?low"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:high|increased|elevated)\s+"
                r"(?:fall\s+risk|risk\s+of\s+falling)|"
                r"fall\s+risk\s+(?:is\s+)?"
                r"(?:high|identified|positive)|"
                r"(?:assessment\s+)?identif(?:y|ies|ied)\s+"
                r"(?:a\s+)?(?:high|increased|elevated)\s+"
                r"risk\s+of\s+falling"
                r")\b"
            ),
        },
        "irrigation_used": {
            "negative": (
                r"\b(?:"
                r"no\s+irrigation\s+(?:was\s+)?"
                r"(?:used|performed)|"
                r"irrigation\s+(?:was\s+)?not\s+"
                r"(?:used|performed|required)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"irrigation\s+(?:was\s+)?"
                r"(?:used|performed|carried\s+out)|"
                r"(?:operative|surgical)\s+field\s+"
                r"(?:was\s+)?"
                r"(?:irrigated|washed|rinsed|flushed)"
                r"(?:\s+with\b)?"
                r")\b"
            ),
        },
        "blood_transfusion": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:blood\s+)?transfusion\s+"
                r"(?:was\s+)?(?:given|administered|required)|"
                r"(?:blood\s+)?transfusion\s+"
                r"(?:was\s+)?not\s+"
                r"(?:given|administered|required)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:a\s+)?(?:blood\s+)?transfusion\s+"
                r"(?:was\s+)?(?:given|administered|performed)|"
                r"(?:the\s+)?patient\s+"
                r"(?:was\s+)?transfused\b"
                r")"
            ),
        },
        "initial_count_waived": {
            "negative": (
                r"\b(?:"
                r"initial\s+(?:surgical\s+)?count\s+"
                r"(?:was\s+)?not\s+waived|"
                r"initial\s+(?:surgical\s+)?count\s+"
                r"(?:was\s+)?(?:performed|completed)"
                r")\b"
            ),
            "positive": (
                r"\binitial\s+(?:surgical\s+)?count\s+"
                r"(?:was\s+)?waived\b"
            ),
        },

        "retained_sponge_case": {
            "negative": (
                r"\b(?:"
                r"no\s+retained\s+sponge\s+"
                r"(?:was\s+)?(?:suspected|identified|found)|"
                r"retained\s+sponge\s+"
                r"(?:was\s+)?not\s+"
                r"(?:suspected|identified|found)"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:a\s+)?retained\s+sponge\s+"
                r"(?:was\s+)?"
                r"(?:suspected|identified|found|present)|"
                r"retained\s+sponge\s+case"
                r")\b"
            ),
        },
        "esu_used": {
            "negative": (
                r"\b(?:"
                r"no\s+(?:electrosurgical\s+unit|esu|"
                r"cautery\s+unit)\s+(?:was\s+)?used|"
                r"(?:electrosurgical\s+unit|esu|"
                r"cautery\s+unit)\s+(?:was\s+)?"
                r"not\s+used"
                r")\b"
            ),
            "positive": (
                r"\b(?:electrosurgical\s+unit|esu|"
                r"cautery\s+unit)"
                r"(?:\s+[A-Za-z0-9-]+)?\s+"
                r"(?:was\s+)?used\b"
            ),
        },
        "outcome_intubated": {
            "negative": (
                r"\b(?:"
                r"(?:the\s+)?patient\s+(?:was\s+)?extubated|"
                r"(?:the\s+)?patient\s+(?:did\s+)?not\s+"
                r"remain\s+intubated|"
                r"(?:the\s+)?patient\s+was\s+not\s+intubated"
                r")\b"
            ),
            "positive": (
                r"\b(?:"
                r"(?:the\s+)?patient\s+(?:remained|was)\s+"
                r"intubated|"
                r"(?:the\s+)?patient\s+was\s+transferred\s+"
                r"(?:while\s+)?intubated|"
                r"transferred\s+intubated"
                r")\b"
            ),
        },
    }

    selected_patterns = field_patterns.get(field_name)

    if selected_patterns:
        # Negative evidence must be checked first because
        # "no count discrepancy" contains "count discrepancy".
        if re.search(
            selected_patterns["negative"],
            normalized_transcript,
            flags=re.IGNORECASE,
        ):
            return "no"

        if re.search(
            selected_patterns["positive"],
            normalized_transcript,
            flags=re.IGNORECASE,
        ):
            return "yes"

    # Generic Boolean wording.
    negative_pattern = (
        r"\b(?:no|false|not confirmed|"
        r"has not been confirmed|"
        r"was not confirmed)\b"
    )

    positive_pattern = (
        r"\b(?:yes|true|confirmed|"
        r"has been confirmed|"
        r"was confirmed)\b"
    )

    # Check generic negative evidence first.
    if re.search(
        negative_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    ):
        return "no"

    if re.search(
        positive_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    ):
        return "yes"

    return None

def extract_person_name(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extract a person's name only when the wording confirms
    the specific patient or staff role in ``metadata``.

    This field-aware routing is important for secondary
    extraction: a valid patient name must not also validate
    a surgeon, assistant, or nurse candidate.
    """

    if metadata.get("field") == "patient_name":
        match = re.search(r"\b(?:enter|record|document)\s+"
                          r"(?P<name>[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
                          r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+){0,3})\s+"
                          r"as\s+(?:the\s+)?patient\s+name\b", transcript, re.I)
        if match:
            return match.group("name").strip()
    # Optional role context can sit between the role and "is".
    # Preserve the stated name; do not repair ASR spellings by guessing.
    transcript = re.sub(r"\b(primary\s+surgeon|lead\s+surgeon|second\s+surgeon|"
                        r"anesthesiologist|scrub\s+nurse|circulating\s+nurse)"
                        r"\s+for\s+(?:this|the)\s+(?:case|operation|procedure)"
                        r"(?=\s+(?:is|was)\b)", r"\1", transcript, flags=re.I)
    transcript = re.sub(r"\bby\s+(?:the\s+)?(anesthesiologist|primary\s+surgeon|scrub\s+nurse|"
                        r"circulating\s+nurse)\s*,\s*(?=(?:doctor|dr\.)\s)",
                        r"\1 is ", transcript, flags=re.I)
    normalized_transcript = transcript.strip()

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    name_pattern = (
        r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
        r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,3}"
    )

    if field_name == "patient_name":
        patterns = [
            (
                rf"\b(?:the\s+)?patient(?:'s)?\s*"
                rf"(?:"
                rf"name\s+(?:is|was)|"
                rf"(?:is|was)(?:\s+named)?|"
                rf"named|:|-"
                rf")\s*"
                rf"(?P<name>{name_pattern})"
                rf"(?="
                rf"\s+and\s+(?:the\s+)?"
                rf"(?:medical\s+record|"
                rf"patient\s+(?:id|identifier|number)|mrn)\b"
                rf"|[.,;:]|$"
                rf")"
            ),
        ]

    else:
        administrative_role_terms = {
            "physician_name": [
                "responsible physician",
                "physician",
            ],
            "recovery_nurse_name": [
                "recovery nurse",
            ],
            "floor_nurse_name": [
                "receiving floor nurse",
                "floor nurse",
            ],
        }

        role_terms = administrative_role_terms.get(
            field_name,
            [
                term.strip()
                for term in metadata.get(
                    "keywords",
                    "",
                ).split(",")
                if term.strip()
            ],
        )

        role_pattern = "(?:" + "|".join(
            re.escape(term).replace(r"\ ", r"\s+")
            for term in sorted(
                role_terms,
                key=len,
                reverse=True,
            )
        ) + ")"

        title_pattern = r"(?:(?:dr\.?|doctor)\s+)?"
        boundary_pattern = r"(?=\s+(?:and|while)\b|[.,;:]|$)"

        patterns = [
            # Role first: "The primary surgeon is Dr. X."
            (
                rf"\b(?:the\s+)?{role_pattern}\b"
                rf"\s*(?:(?:is|was|:|-)\s*)?"
                rf"(?!and\b){title_pattern}"
                rf"(?P<name>{name_pattern})"
                rf"{boundary_pattern}"
            ),
            # Name first: "Dr. X is the primary surgeon."
            (
                rf"\b(?!and\b|while\b){title_pattern}"
                rf"(?P<name>{name_pattern})"
                rf"\s+(?:is|was)\s+(?:the\s+)?"
                rf"{role_pattern}\b"
                rf"{boundary_pattern}"
            ),
        ]
        relief_base_roles = {
            "relief_scrub_nurse": (
                r"scrub\s+nurse"
            ),
            "relief_circulating_nurse": (
                r"circulating\s+nurse"
            ),
        }

        base_role_pattern = (
            relief_base_roles.get(field_name)
        )

        if base_role_pattern is not None:
            patterns.insert(
                0,
                (
                    rf"\b(?:the\s+)?"
                    rf"{base_role_pattern}\b"
                    rf"\s*(?:(?:is|was|:|-)\s*)?"
                    rf"{title_pattern}"
                    rf"(?:{name_pattern})"
                    rf"\s*,?\s+"
                    rf"(?:was\s+)?relieved\s+by\s+"
                    rf"{title_pattern}"
                    rf"(?P<name>{name_pattern})"
                    rf"{boundary_pattern}"
                ),
            )

        if field_name == "surgeon_1":
            patterns.append(
                (
                    rf"\bperformed\s+by\s+"
                    rf"{title_pattern}"
                    rf"(?P<name>{name_pattern})"
                    rf"{boundary_pattern}"
                )
            )
            patterns.append(
                (
                    rf"\b(?:dr\.?|doctor)\s+"
                    rf"(?P<name>{name_pattern})\s+"
                    rf"performed\s+(?:an?|the)\s+"
                )
            )

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is not None:
            name = match.group("name").strip()

            invalid_edge_words = {
                "and",
                "doctor",
                "dr",
                "is",
                "the",
                "was",
            }

            name_words = [
                word.strip(".").lower()
                for word in name.split()
            ]

            if (
                not name_words
                or name_words[0] in invalid_edge_words
                or name_words[-1] in invalid_edge_words
            ):
                continue

            return name

    return None

def extract_categorical(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extracts and normalizes one value from the
    field's configured list of allowed values.
    """

    if metadata.get("field") == "transfer_destination":
        if re.search(r"\b(?:transferred|sent|destination\s+(?:is|was))\s+"
                     r"(?:to\s+)?(?:the\s+)?"
                     r"(?:(?:surgical|medical|general|inpatient)\s+)?ward\b",
                     transcript, re.I):
            return "floor"
    normalized_transcript = transcript.strip().lower()

    allowed_values = [
        value.strip().lower()
        for value in metadata.get(
            "allowed_values",
            "",
        ).split("|")
        if value.strip()
    ]

    for canonical_value in sorted(
        allowed_values,
        key=len,
        reverse=True,
    ):
        spoken_value = re.escape(
            canonical_value
        ).replace("_", r"[-\s_]+")

        pattern = CATEGORY_PATTERNS.get(
            canonical_value,
            rf"\b{spoken_value}\b",
        )

        if re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        ):
            return canonical_value

    return None

def extract_identifier(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """Extract an identifier only for the requested field type."""
    identifier_pattern = (
        r"[A-Za-z0-9]+"
        r"(?:[-_/][A-Za-z0-9]+)*"
    )

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    field_prefixes = {
        "patient_id": (
            r"(?:patient\s*(?:id|identifier|number)|"
            r"medical\s+record\s+(?:number|no\.?|#)|"
            r"mrn|hospital\s+number)"
        ),
        "case_id": (
            r"(?:"
            r"case\s*(?:id|number)|"
            r"(?:surgical|operative|operation)\s+case"
            r"(?:\s+(?:id|number))?"
            r")"
        ),
        "or_room_number": (
            r"(?:"
            r"or(?:\s+room)?|"
            r"operating\s+(?:room|theatre)|"
            r"surgical\s+suite|"
            r"room|theatre"
            r")"
            r"\s*(?:number|no\.?|#)?"
        ),
        "operation_code": (
            r"(?:operation|procedure|surgical)\s+code"
        ),
        "tourniquet_serial_number": (
            r"(?:tourniquet\s+(?:serial|device)\s+number)"
        ),
        "esu_serial_number": (
            r"(?:"
            r"esu\s+serial\s+number|"
            r"electrosurgical\s+unit\s+serial\s+number|"
            r"cautery\s+(?:unit\s+)?serial\s+number|"
            r"esu|"
            r"electrosurgical\s+unit|"
            r"cautery\s+unit"
            r")"
        ),
    }

    field_prefix = field_prefixes.get(field_name)

    # Processing-metadata identifiers are populated by the
    # system, not extracted from clinical speech.
    if field_prefix is None:
        return None

    pattern = (
        rf"\b{field_prefix}\b"
        rf"\s*(?:is|was|:|-)?\s*"
        rf"(?P<identifier>{identifier_pattern})\b"
    )

    normalized_transcript = transcript.strip()

    match = re.search(
        pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if match is not None:
        identifier = match.group(
            "identifier"
        )

        # Equipment serial numbers must contain
        # at least one digit.
        if field_name in {
            "tourniquet_serial_number",
            "esu_serial_number",
        }:
            if not re.search(
                r"\d",
                identifier,
            ):
                return None

        # A time statement such as "entered the
        # operating room at 07:15" must not treat
        # "at" as an operating-room identifier.
        if field_name == "or_room_number":
            invalid_room_identifiers = {
                "at",
                "for",
                "in",
                "is",
                "on",
                "the",
                "was",
            }

            if (
                identifier.lower()
                in invalid_room_identifiers
            ):
                return None

        return identifier

    return None

def extract_allergy_list(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """Extracts and normalizes stated patient allergies."""

    # Reporting an allergy is an affirmative statement, unlike denying it.
    match = re.search(r"\b(?:the\s+)?patient\s+(?:reports?|reported)\s+"
                      r"(?:a|an|documented|known)?\s*allerg(?:y|ies)\s+to\s+"
                      r"(?P<value>[^.;]+)", transcript, re.I)
    if match and not re.search(r"\b(?:no|not|denies|denied|possible|suspected)\b",
                               match.group("value"), re.I):
        return "|".join(x.strip() for x in re.split(r",|\band\b", match.group("value"), flags=re.I) if x.strip())
    normalized_transcript = transcript.strip()

    if re.search(
        r"\b(?:no known (?:drug\s+)?allergies|"
        r"no (?:drug\s+)?allergies|nkda)\b",
        normalized_transcript,
        flags=re.IGNORECASE,
    ):
        return "none"

    patterns = [
        (
            r"\b(?:the\s+)?patient\s+is\s+allergic\s+to\s+"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),
        (
            r"\b(?:known\s+)?allerg(?:y|ies)\s*"
            r"(?:are|is|include|:|-)\s*"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),
        (
            r"\b(?:the\s+patient|patient|he|she|they)\s+"
            r"(?:has|have)\s+"
            r"(?:a\s+)?(?:documented\s+|known\s+)?"
            r"allerg(?:y|ies)\s+to\s+"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),

    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is not None:
            value = match.group(
                "value"
            ).strip(" ,")

            # Normalize natural allergy separators to the
            # canonical pipe delimiter expected by the
            # configured list converter.
            allergy_items = re.split(
                r"\s*(?:,|;|\band\b)\s*",
                value,
                flags=re.IGNORECASE,
            )

            normalized_items = [
                item.strip()
                for item in allergy_items
                if item.strip()
            ]

            if not normalized_items:
                continue

            return "|".join(
                normalized_items
            )

    return None
def extract_time(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extracts a field-specific operative time and
    normalizes it to HH:MM.
    """

    # Convert a clock followed by an explicit day-period phrase, not any
    # arbitrary mention of afternoon elsewhere in a multi-sentence text.
    transcript = re.sub(r"(\b\d{1,2}(?:[:.]\d{2})?)\s+in\s+the\s+"
                        r"(morning|afternoon|evening)\b",
                        lambda m: m.group(1) + (" am" if m.group(2).lower() == "morning" else " pm"),
                        transcript, flags=re.I)

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    event_patterns = {
        "patient_in_time": (
            r"\b(?:patient in time|"
            r"(?:the\s+)?patient\s+"
            r"(?:entered|arrived\s+(?:in|at)|"
            r"was\s+brought\s+(?:into|to))\s+"
            r"(?:the\s+)?(?:operating\s+(?:room|suite)|or))\b"
        ),
        "patient_out_time": (
            r"\b(?:patient out time|"
            r"(?:the\s+)?patient\s+"
            r"(?:left|exited|departed)\s+"
            r"(?:the\s+)?(?:operating\s+(?:room|suite)|or|room))\b"
        ),
        "anesthesia_induction_time": (
            r"\b(?:anesthe(?:sia|tic) induction time|"
            r"anesthe(?:sia|tic) induction\s+"
            r"(?:began|started|occurred|was\s+performed)|"
            r"induction\s+(?:began|started|occurred))\b"
        ),
        "incision_time": (
            r"\b(?:incision time|skin incision time|"
            r"(?:first\s+)?(?:skin\s+)?incision\s+"
            r"(?:started|began|was\s+made|"
            r"was\s+performed|took\s+place|commenced))\b"
        ),
        "surgery_end_time": (
            r"\b(?:surgery end time|end of surgery|"
            r"surgery\s+(?:was\s+)?(?:ended|finished|completed|concluded)|"
            r"operation\s+(?:was\s+)?(?:ended|finished|completed|concluded)|"
            r"procedure\s+(?:was\s+)?(?:ended|finished|completed|concluded))\b"
        ),
        "dressing_time": (
            r"\b(?:dressing time|"
            r"(?:the\s+)?(?:surgical\s+)?dressing\s+"
            r"(?:was\s+)?(?:applied|completed|finished))\b"
        ),
    }

    event_pattern = event_patterns.get(field_name)

    if event_pattern is None:
        return None

    clock_pattern = (
        r"(?P<hour>\d{1,2})"
        r"(?:[:.](?P<minute>[0-5]\d))?"
        r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)?"
    )

    pattern = (
        event_pattern
        + r"[^\d]{0,30}"
        + clock_pattern
        + r"(?![:.]\d)"
        + r"(?=\s|[.,;]|$)"
    )

    match = re.search(
        pattern,
        transcript.strip(),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    hour = int(match.group("hour"))
    minute = int(
        match.group("minute") or 0
    )

    meridiem = match.group("meridiem")

    if meridiem:
        meridiem = (
            meridiem.lower().replace(".", "")
        )

        if not 1 <= hour <= 12:
            return None

        if meridiem == "pm" and hour != 12:
            hour += 12

        elif meridiem == "am" and hour == 12:
            hour = 0

    elif not 0 <= hour <= 23:
        return None

    return f"{hour:02d}:{minute:02d}"
def extract_event_text(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """Extracts a stated unexpected event or complication."""
    normalized_transcript = transcript.strip()

    no_event_pattern = (
        r"\b(?:no unexpected events?|"
        r"no complications?|"
        r"no adverse events?)\b"
    )
    if field_name == "unexpected_event":
        count_issue_pattern = (
            r"\b(?:"
            r"(?:final\s+)?count\s+discrepancy|"
            r"(?:sponge|instrument|needle|final|surgical)\s+"
            r"count\s+(?:was\s+)?incorrect|"
            r"counts?\s+(?:was|were)\s+incorrect|"
            r"incorrect\s+(?:sponge\s+|instrument\s+|"
            r"needle\s+)?count|"
            r"discrepancy.{0,60}\b"
            r"(?:final\s+)?(?:surgical\s+)?count"
            r")\b"
        )

        count_issue_match = re.search(
            count_issue_pattern,
            normalized_text,
            flags=re.IGNORECASE,
        )

        if count_issue_match is not None:
            count_actions = extract_count_actions(
                transcript,
                {
                    "field": "count_actions",
                },
            )

            # A standalone count issue belongs in the
            # dedicated count-discrepancy field. It becomes
            # an unexpected event only when a corrective
            # response is explicitly documented.
            if not count_actions:
                return None

            return [event_text]

    if re.search(
        no_event_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    ):
        return "none"

    patterns = [
        (
            r"\b(?:an?\s+)?unexpected event\s+occurred"
            r"\s*(?::|-)?\s*"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),
        (
            r"\b(?:a\s+)?complication\s+occurred"
            r"\s*(?::|-)?\s*"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),
        (
            r"\b(?:an?\s+)?adverse event\s+occurred"
            r"\s*(?::|-)?\s*"
            r"(?P<value>.+?)(?=[.;]|$)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is not None:
            return match.group("value").strip(" ,:-")

    return None
def extract_procedure_text(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extract a procedure narrative only when the clause
    explicitly describes a performed procedure, procedural
    completion, conversion, cancellation, or change of
    surgical approach.
    """

    # A bare end-time statement does not describe what was performed.
    if re.fullmatch(r"\s*(?:the\s+)?(?:surgery|operation|procedure)\s+"
                    r"(?:was\s+)?(?:completed|ended|finished|concluded)\s+"
                    r"at\s+\d{1,2}(?:[:.]\d{2})?"
                    r"(?:\s*(?:a\.?m\.?|p\.?m\.?)|\s+in\s+the\s+"
                    r"(?:morning|afternoon|evening))?\s*[.!]?\s*", transcript, re.I):
        return None

    cleaned_transcript = transcript.strip(" .,:;-")

    if not cleaned_transcript:
        return None
    approach_pattern = (
        r"(?:laparoscopic|open|robotic|endoscopic|"
        r"minimally\s+invasive)"
    )

    conversion_patterns = [
        # Unexpected bleeding required conversion from
        # the laparoscopic procedure to an open procedure.
        (
            r"\b(?:required|necessitated|underwent)\s+"
            r"(?:a\s+)?conversion\s+from\s+"
            r"(?:the\s+)?"
            rf"(?P<source>{approach_pattern})"
            r"(?:\s+procedure)?\s+to\s+"
            r"(?:(?:a|an|the)\s+)?"
            rf"(?P<target>{approach_pattern})"
            r"(?:\s+procedure)?\b"
        ),

        # The laparoscopic procedure was converted to
        # an open procedure.
        (
            r"\b(?:the\s+)?"
            rf"(?P<source>{approach_pattern})\s+"
            r"procedure\s+(?:was\s+)?converted\s+to\s+"
            r"(?:(?:a|an|the)\s+)?"
            rf"(?P<target>{approach_pattern})"
            r"(?:\s+procedure)?\b"
        ),

        # The case was converted from laparoscopic to
        # open procedure.
        (
            r"\bconverted\s+from\s+"
            r"(?:the\s+)?"
            rf"(?P<source>{approach_pattern})"
            r"(?:\s+procedure)?\s+to\s+"
            r"(?:(?:a|an|the)\s+)?"
            rf"(?P<target>{approach_pattern})"
            r"(?:\s+procedure)?\b"
        ),
    ]

    for conversion_pattern in conversion_patterns:
        conversion_match = re.search(
            conversion_pattern,
            cleaned_transcript,
            flags=re.IGNORECASE,
        )

        if conversion_match is None:
            continue

        source_approach = (
            conversion_match.group("source")
            .strip()
            .lower()
        )

        target_approach = (
            conversion_match.group("target")
            .strip()
            .lower()
        )

        return (
            f"converted from {source_approach} "
            f"to {target_approach} procedure"
        )

    procedure_pattern = (
        r"\b(?:procedure|operation|surgery|"
        r"surgical\s+(?:procedure|approach|plan))\b"
        r".*\b(?:performed|completed|converted|"
        r"changed|modified|aborted|cancelled|"
        r"finished|continued)\b"
        r"|"
        r"\b(?:converted|changed|modified)\b"
        r".*\b(?:procedure|operation|"
        r"surgical\s+(?:approach|plan))\b"
    )

    if not re.search(
        procedure_pattern,
        cleaned_transcript,
        flags=re.IGNORECASE,
    ):
        return None

    return cleaned_transcript
def extract_text(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extract a clean textual value according to the
    predicted field.

    Introductory clinical wording is removed when a
    field-specific pattern is available.
    """

    field = metadata.get("field", "").strip().lower()
    text = transcript.strip(" .,:;-")
    # A measured quantity is not a named operation merely because the
    # sentence contains "procedure was".
    if field == "procedure_name" and re.search(
        r"\b(?:blood\s+loss|blood\s+lost|fluid\s+volume|temperature|"
        r"blood\s+pressure|heart\s+rate)\b", text, re.I
    ) and not re.search(r"\b(?:planned\s+procedure|procedure\s+name|"
                        r"operation\s+performed)\b", text, re.I):
        return None
    if field in {"preoperative_diagnosis", "postoperative_diagnosis"}:
        relation = "before" if field == "preoperative_diagnosis" else "after"
        match = re.search(
            rf"\bdiagnosis\s+(?:(?:established|documented|recorded|confirmed|"
            rf"identified)\s+)?{relation}\s+(?:the\s+)?"
            rf"(?:surgery|operation|procedure)\s+(?:is|was|:|-)\s*"
            rf"(?P<value>[^.;]+)", text, re.I)
        if match:
            return re.sub(r"^(?:a|an|the)\s+", "", match.group("value"), flags=re.I).strip()
    if field == "procedure_site":
        match = re.search(r"\b(?:procedure|surgical|operation)\s+site\s+"
                          r"(?:is|was|:|-)\s*(?:the\s+)?"
                          r"(?P<value>.+?)(?=\s*(?:,?\s+and\s+(?:the\s+)?"
                          r"(?:laterality|procedure|anesthesia)\b|[.;]|$))", text, re.I)
        if match:
            return re.sub(r"^(?:left|right|bilateral)\s+", "",
                          match.group("value").strip(), flags=re.I)
    if field == "other_destination" and re.search(
        r"\b(?:transferred|sent)\s+to\s+(?:the\s+)?"
        r"(?:(?:surgical|medical|general|inpatient)\s+)?ward\b", text, re.I):
        return None

    cleaned_transcript = (
        transcript.strip(" .,:;-")
    )

    if not cleaned_transcript:
        return None

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    field_patterns = {
        "preoperative_diagnosis": [
            (
                r"\b(?:the\s+)?"
                r"(?:preoperative|pre-operative|pre-op)\s+"
                r"diagnosis\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\bdiagnosis before surgery\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
        ],
        "postoperative_diagnosis": [
            (
                r"\b(?:following|after)\s+"
                r"(?:the\s+)?(?:operation|surgery|procedure)\s*,?\s*"
                r"(?:the\s+)?diagnosis\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:the\s+)?"
                r"(?:postoperative|post-operative|post-op)\s+"
                r"diagnosis\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\bdiagnosis after surgery\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
        ],
        "procedure_name": [
            (
                r"\b(?:the\s+)?planned\s+"
                r"(?:procedure|operation)\s+"
                r"(?:is|was|:|-)\s*"
                r"(?:(?:a|an|the)\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:the\s+)?procedure\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:the\s+)?operation performed\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\bsurgical procedure\s+"
                r"(?:is|was|:|-)\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
                        (
                r"\b(?:the\s+)?procedure performed\s+"
                r"(?:is|was|:|-)\s*(?:a|an|the)?\s*"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:dr\.?|doctor)\s+"
                r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
                r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,3}\s+"
                r"performed\s+(?:an?|the)\s+"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
                ],
        "procedure_site": [
            (   r"\b(?:dr\.?|doctor)\s+"
                r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
                r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+){1,3}\s+"
                r"performed\s+(?:an?|the)\s+.+?\s+"
                r"(?:of|on)\s+(?:the\s+)?"
                r"(?P<value>(?:left|right|bilateral)\s+.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:the\s+)?surgeon\s+performed\s+"
                r"(?:an?|the)\s+.+?\s+(?:of|on)\s+(?:the\s+)?"
                r"(?P<value>(?:left|right|bilateral)\s+.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:procedure|surgical|operation)\s+site\s+"
                r"(?:is|was|:|-)\s*(?:the\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\bplanned\s+(?:procedure|operation)\s+"
                r"(?:is|was|:|-)\s*.+?\s+"
                r"(?:of|on)\s+(?:the\s+)?"
                r"(?P<value>(?:left|right)\s+.+?)(?=[.;]|$)"
            ),
        ],
        "other_position": [
            (
                r"\b(?:the\s+)?"
                r"(?:operative|patient)\s+position\s+"
                r"(?:is|was|:|-)\s*"
                r"(?:(?:a|an|the)\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:the\s+)?patient\s+"
                r"(?:was\s+)?positioned\s+"
                r"(?:in|as)\s+"
                r"(?:(?:a|an|the)\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:before|after)\s+transfer\s+to\s+(?:the\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
        ],
        "external_lab_name": [
        (
            r"\b(?:the\s+)?(?:specimen|sample)\s+"
            r"(?:was\s+)?"
            r"(?:sent|transferred|submitted)\s+to\s+"
            r"(?:the\s+)?"
            r"(?P<value>"
            r"(?:[A-Za-z][A-Za-z'-]*\s+){1,5}"
            r"(?:Laboratory|Lab)"
            r")\b"
        ),
        ],
        "other_destination": [
            (
                r"\b(?:"
                r"(?:the\s+)?patient\b.{0,80}?\b"
                r")?"
                r"(?:was\s+)?transferred\s+to\s+"
                r"(?:the\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
            (
                r"\b(?:before|after)\s+transfer\s+to\s+"
                r"(?:the\s+)?"
                r"(?P<value>.+?)(?=[.;]|$)"
            ),
        ],

    }

    patterns = field_patterns.get(
        field_name,
        [],
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            transcript,
            flags=re.IGNORECASE,
        )

        if match is not None:
            value = match.group(
                "value"
            ).strip(" .,:;-")

            if not value:
                continue

            if field_name == "other_destination":
                standard_destination_pattern = (
                    r"^(?:"
                    r"pacu|"
                    r"post[-\s]?anesthesia\s+care\s+unit|"
                    r"post[-\s]?anaesthesia\s+care\s+unit|"
                    r"icu|"
                    r"intensive\s+care\s+unit|"
                    r"ccu|"
                    r"(?:coronary|cardiac)\s+care\s+unit|"
                    r"(?:the\s+)?floor"
                    r")$"
                )

                if re.fullmatch(
                    standard_destination_pattern,
                    value,
                    flags=re.IGNORECASE,
                ):
                    continue

            # Diagnosis fields store the clinical diagnosis
            # without a leading grammatical article.
            if field_name in {
                "preoperative_diagnosis",
                "postoperative_diagnosis",
            }:
                value = re.sub(
                    r"^(?:a|an|the)\s+",
                    "",
                    value,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()

            # Store laterality separately instead of
            # duplicating it inside the procedure site.
            elif field_name == "procedure_site":
                value = re.sub(
                    r"^(?:the\s+)?"
                    r"(?:left|right|bilateral)\s+",
                    "",
                    value,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()

            # When an explicitly lateralized anatomical
            # site follows "of" or "on", keep that site in
            # procedure_site and retain only the procedure
            # itself here.
            elif field_name == "procedure_name":
                if re.match(
                    r"^(?:completed|finished|continued|"
                    r"converted|changed|modified|aborted|"
                    r"cancelled)\b",
                    value,
                    flags=re.IGNORECASE,
                ):
                    continue
                value = re.sub(
                    r"\s+(?:of|on)\s+(?:the\s+)?"
                    r"(?:left|right|bilateral)\s+"
                    r"[a-z][a-z\s-]*$",
                    "",
                    value,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()

            if value:
                return value

    # A field with configured patterns must not fall back
    # to accepting an unrelated complete sentence.
    if field_name in field_patterns:
        return None

    # Only genuinely narrative fields without configured
    # patterns may retain the complete statement.
    return cleaned_transcript

def extract_categorical_list(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extracts one or more configured categorical values.

    Multiple values are temporarily joined with "|"
    so they can later be converted into a Python list.
    """

    normalized_transcript = (
        transcript.strip().lower()
    )
    field_name = metadata.get(
    "field",
    "",
    ).strip().lower()

    if field_name == "laboratory_requests":
        test_patterns = [
            (
                "microbiology culture",
                r"\bmicrobiology\s+culture\b",
            ),
            (
                "histopathology",
                r"\bhistopatholog(?:y|ical)\b",
            ),
            (
                "pathology",
                r"\bpathology\b",
            ),
            (
                "cytology",
                r"\bcytology\b",
            ),
            (
                "frozen section",
                r"\bfrozen\s+section\b",
            ),
            (
                "culture",
                r"\bculture\b",
            ),
        ]

        requested_tests = []

        for canonical_value, pattern in test_patterns:
            if not re.search(
                pattern,
                normalized_transcript,
                flags=re.IGNORECASE,
            ):
                continue

            # Do not add generic "culture" when the more
            # specific "microbiology culture" was found.
            if (
                canonical_value == "culture"
                and "microbiology culture"
                in requested_tests
            ):
                continue

            requested_tests.append(
                canonical_value
            )

        if requested_tests:
            return "|".join(requested_tests)

    allowed_values = [
        value.strip().lower()
        for value in metadata.get(
            "allowed_values",
            "",
        ).split("|")
        if value.strip()
    ]

    category_patterns = {
        "general": (
            r"\bgeneral(?:\s+anesthesia)?\b"
        ),
        "spinal": (
            r"\bspinal(?:\s+anesthesia)?\b"
        ),
        "epidural": (
            r"\bepidural(?:\s+anesthesia)?\b"
        ),
        "local": (
            r"\blocal(?:\s+anesthesia)?\b"
        ),
        "regional": (
            r"\bregional(?:\s+anesthesia)?\b"
        ),
        "iv_sedation": (
            r"\b(?:iv|intravenous)\s+sedation\b"
        ),
        "monitored": (
            r"\b(?:monitored anesthesia care|"
            r"monitored anesthesia|mac)\b"
        ),
    }

    extracted_values = []

    for canonical_value in allowed_values:
        pattern = category_patterns.get(
            canonical_value,
            rf"\b{re.escape(canonical_value)}\b",
        )

        if re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        ):
            extracted_values.append(
                canonical_value
            )

    if not extracted_values:
            return None

    return "|".join(extracted_values)

def extract_medication_details(
    transcript: str,
    metadata: dict[str, str],
) -> list[dict[str, str]] | None:
    """
    Extract one administered medication with its dose.

    The function only returns a medication when both
    the drug name and dose are explicitly present.
    """

    normalized_transcript = transcript.strip().lower()

    number_pattern = (
        r"(?:\d+(?:\.\d+)?|one|two|three|four|five|"
        r"six|seven|eight|nine|ten)"
    )

    unit_pattern = (
        r"(?:grams?|g|milligrams?|mg|micrograms?|mcg|"
        r"milliliters?|ml)"
    )

    patterns = [
        # Antibiotic prophylaxis was provided with
# 900 mg of a medication.
        (
            r"\b(?:antibiotic(?:s)?\s+prophylaxis|"
            r"prophylactic\s+antibiotic(?:s)?)\s+"
            r"(?:was\s+)?(?:provided|given|administered)\s+"
            r"with\s+"
            rf"(?P<number>{number_pattern})\s*"
            rf"(?P<unit>{unit_pattern})\s+of\s*"
            rf"(?P<drug>[a-z][a-z-]*)\b"
        ),
        # One gram of cefazolin was administered.
        (
            rf"\b(?P<number>{number_pattern})\s*"
            rf"(?P<unit>{unit_pattern})\s+of\s*"
            rf"(?P<drug>[a-z][a-z-]*)\s+"
            rf"(?:(?:was|were)\s+)?"
            rf"(?:also\s+)?"
            rf"(?:administered|given)\b"
        ),

        # Cefazolin one gram was administered.
        (
            rf"\b(?P<drug>[a-z][a-z-]*)\s+"
            rf"(?P<number>{number_pattern})\s*"
            rf"(?P<unit>{unit_pattern})\s+"
            rf"(?:(?:was|were)\s+)?(?:administered|given)\b"
        ),

        # The patient received ondansetron four milligrams.
        (
            rf"\b(?:the\s+)?patient\s+received\s+"
            rf"(?P<drug>[a-z][a-z-]*)\s+"
            rf"(?P<number>{number_pattern})\s*"
            rf"(?P<unit>{unit_pattern})\b"
        ),
    ]

    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    unit_normalization = {
        "g": "gram",
        "gram": "gram",
        "grams": "grams",
        "mg": "milligram",
        "milligram": "milligram",
        "milligrams": "milligrams",
        "mcg": "microgram",
        "microgram": "microgram",
        "micrograms": "micrograms",
        "ml": "milliliter",
        "milliliter": "milliliter",
        "milliliters": "milliliters",
    }

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        number = match.group("number").lower()
        number = number_words.get(number, number)

        unit = match.group("unit").lower()
        unit = unit_normalization.get(unit, unit)
        # Canonicalize units according to quantity.
        try:
            numeric_dose = float(number)
        except ValueError:
            numeric_dose = None

        singular_to_plural = {
            "gram": "grams",
            "milligram": "milligrams",
            "microgram": "micrograms",
            "milliliter": "milliliters",
        }

        if (
            numeric_dose is not None
            and numeric_dose != 1
        ):
            unit = singular_to_plural.get(
                unit,
                unit,
            )

        drug = match.group("drug").lower()


        return [
            {
                "name": drug,
                "dose": f"{number} {unit}",
            }
        ]

    return None

def extract_specimen_details(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extract the stated identity or type of a specimen.

    Examples:
        appendix specimen
        gallbladder specimen
        lymph node sample
    """

    normalized_transcript = (
        transcript.strip().lower()
    )
    segment_specimen_pattern = (
        r"\b(?:a\s+|an\s+|the\s+)?"
        r"(?:segment|portion|piece|sample)\s+of\s+"
        r"(?P<specimen>"
        r"[a-z][a-z-]*"
        r"(?:\s+(?!was\b|were\b)"
        r"[a-z][a-z-]*){0,3}"
        r")\s+"
        r"(?:was\s+)?"
        r"(?:collected|obtained|submitted)\s+"
        r"as\s+(?:a\s+)?(?:specimen|sample)\b"
    )

    segment_match = re.search(
        segment_specimen_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if segment_match is not None:
        specimen = segment_match.group(
            "specimen"
        ).strip()

        return [specimen]
    source_pattern = (
        r"\b(?:the\s+|a\s+|an\s+)?"
        r"(?P<specimen_type>"
        r"[a-z][a-z-]*"
        r"(?:\s+[a-z][a-z-]*){0,2}"
        r")\s+"
        r"(?:specimen|sample)\s+from\s+"
        r"(?:the\s+)?"
        r"(?P<source>"
        r"[a-z][a-z-]*"
        r"(?:\s+(?!was\b|were\b)"
        r"[a-z][a-z-]*){0,3}"
        r")\s+"
        r"(?:(?:was|were)\s+)?"
        r"(?:collected|obtained|sent|submitted)\b"
    )

    source_match = re.search(
        source_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if source_match is not None:
        specimen_type = source_match.group(
            "specimen_type"
        ).strip()
        source = source_match.group(
            "source"
        ).strip()

        return [f"{source} {specimen_type}"]

    patterns = [
        (
            r"\b(?:the\s+|a\s+|an\s+)?"
            r"(?P<specimen>"
            r"[a-z][a-z-]*"
            r"(?:\s+[a-z][a-z-]*){0,4}"
            r")\s+"
            r"(?:specimen|sample)\s+"
            r"(?:was\s+)?"
            r"(?:collected|obtained|sent|submitted)\b"
        ),
        (
            r"\b(?:specimen|sample)\s+"
            r"(?:was\s+)?"
            r"(?:identified|described)\s+as\s+"
            r"(?P<specimen>"
            r"[a-z][a-z-]*"
            r"(?:\s+[a-z][a-z-]*){0,4}"
            r")"
            r"(?=[.,;:]|$)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        specimen = match.group(
                "specimen"
            ).strip(" ,.-")

        invalid_specimen_values = {
                "a",
                "an",
                "the",
            }

        if (
                specimen
                and specimen.lower()
                not in invalid_specimen_values
            ):
                return [specimen]

        return None


def extract_narrative_list(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """
    Extract narrative findings stored in a list field.

    The parser accepts the text only when an explicit
    findings phrase is present.
    """

    normalized_transcript = (
        transcript.strip().rstrip(" .;")
    )

    if not normalized_transcript:
        return None
    field_name = metadata.get(
    "field",
    "",
    ).strip().lower()

    if field_name == "specimen_other_notes":
        note_patterns = [
            (
                "container marked urgent",
                r"\b(?:the\s+)?container\s+"
                r"(?:was\s+)?marked\s+urgent\b",
            ),
            (
                "transported on ice",
                r"\b(?:was\s+)?transported\s+"
                r"on\s+ice\b",
            ),
            (
                "container labeled",
                r"\b(?:the\s+)?container\s+"
                r"(?:was\s+)?labeled\b",
            ),
            (
                "fresh specimen",
                r"\bfresh\s+specimen\b",
            ),
        ]

        notes = []

        for canonical_note, pattern in note_patterns:
            if re.search(
                pattern,
                normalized_transcript,
                flags=re.IGNORECASE,
            ):
                notes.append(
                    canonical_note
                )

        if notes:
            return "|".join(notes)

        return None

    prefix_pattern = (
        r"^(?:the\s+)?"
        r"(?:"
        r"(?:(?:operative|procedure|surgical)\s+)?"
        r"findings?|"
        r"operative\s+(?:inspection|examination)"
        r")\s*"
        r"(?:showed|show|revealed|reveal|included|"
        r"include|demonstrated|demonstrate|were|are|"
        r"was|is|:|-)\s*"
    )

    # Confirm that an explicit findings prefix exists.
    prefix_match = re.search(
        prefix_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if prefix_match is None:
        return None

    value = re.sub(
        prefix_pattern,
        "",
        normalized_transcript,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")

    # Normalize a leading grammatical article.
    value = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

    if not value:
        return None

    # Keep phrases containing "without" together.
    items = re.split(
        r"\s*(?:,|;|\band\b)\s*",
        value,
        flags=re.IGNORECASE,
    )

    cleaned_items = [
        item.strip(" ,;:-")
        for item in items
        if item.strip(" ,;:-")
    ]

    if not cleaned_items:
        return None

    return "|".join(cleaned_items)

def extract_drain_details(
        transcript: str,
        metadata: dict[str,str],
)->list[str] | None:
    normalized_transcript= transcript.strip()
    if not normalized_transcript:
        return None
    if re.search(
        r"\bjackson[-\s]?pratt\s+"
        r"(?:drain|train)\b",
        normalized_transcript,
        flags=re.IGNORECASE,
    ):
        return ["Jackson-Pratt drain"]

    device_pattern= (
        r"(?P<device>"
        r"(?:(?!(?:a|an|the)\b)"
        r"[a-z][a-z-]*\s+){1,3}"
        r"(?:drain|train|catheter|tube|line)"
        r")"
    )
    placement_pattern = (
        r"(?:was\s+)?"
        r"(?:inserted|placed|positioned|"
        r"left\s+in\s+place)"
    )
    pattern = (
        r"\b(?:a|an|the)?\s*"
        rf"{device_pattern}\s+"
        rf"{placement_pattern}\b"
    )
    match = re.search(
        pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    device = match.group("device").strip().lower()

    device_normalization = {
        "jackson pratt drain": "Jackson-Pratt drain",
        "jackson-pratt drain": "Jackson-Pratt drain",
        "jp drain": "Jackson-Pratt drain",
        "jackson pratt train": "Jackson-Pratt drain",
    }

    normalized_device = device_normalization.get(
        device,
        device,
    )

    return [normalized_device]

def extract_implant_details(
    transcript: str,
    metadata: dict[str, str],
) -> list[dict[str, str]] | None:
    normalized_transcript = transcript.strip()

    if not normalized_transcript:
        return None
    implant_type_pattern = (
        r"(?P<implant_type>"
        r"(?:(?!(?:a|an|the)\b)"
        r"[a-z0-9][a-z0-9-]*\s+){1,3}"
        r"(?:plate|screw|mesh|prosthesis|implant|"
        r"stent|graft|device)"
        r")"
    )
    identifier_label_pattern = (
        r"(?:serial|registry|lot)\s+"
        r"(?:number|no\.?)\s*"
    )

    identifier_pattern = (
        r"(?P<serial_number>[a-z0-9-]+)"
    )
    placement_pattern = (
        r"(?:implanted|inserted|placed)"
    )
    patterns = [
        # A stainless steel fixation plate bearing
        # serial number SF6291 was implanted.
        (
            r"\b(?:a|an|the)?\s*"
            rf"{implant_type_pattern}\s+"
            r"(?:bearing|carrying)\s+"
            rf"{identifier_label_pattern}"
            rf"{identifier_pattern}\s+"
            r"(?:was\s+)?"
            rf"{placement_pattern}\b"
        ),
        # A titanium plate with serial number TP842
        # was implanted.
        (
            r"\b(?:a|an|the)?\s*"
            rf"{implant_type_pattern}\s+with\s+"
            rf"{identifier_label_pattern}"
            rf"{identifier_pattern}\s+"
            r"(?:was\s+)?"
            rf"{placement_pattern}\b"
        ),

        # A titanium plate was implanted with serial
        # number TP842.
        (
            r"\b(?:a|an|the)?\s*"
            rf"{implant_type_pattern}\s+"
            r"(?:was\s+)?"
            rf"{placement_pattern}\s+with\s+"
            rf"{identifier_label_pattern}"
            rf"{identifier_pattern}\b"
        ),
    ]
    for pattern in patterns:
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue
        implant_type = (
            match.group("implant_type")
            .strip()
            .lower()
        )
        implant_type = re.sub(
            r"\bstainless\s+steel\b",
            "stainless-steel",
            implant_type,
            flags=re.IGNORECASE,
        )

        serial_number = (
            match.group("serial_number")
            .strip()
            .upper()
        )

        return [
            {
                "type": implant_type,
                "serial_number": serial_number,
            }
        ]

    return None
def extract_count_stage_details(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extract explicitly completed surgical-count stages.
    """
    normalized_transcript = transcript.strip().lower()

    completion_pattern = (
        r"\b(?P<stages>"
        r"(?:initial|first|second|closing|final)"
        r"(?:\s*(?:,|and)\s*"
        r"(?:initial|first|second|closing|final))*"
        r")\s+counts?\s+"
        r"(?:(?:was|were)\s+)?"
        r"(?:completed|performed|conducted)\b"
    )

    match = re.search(
        completion_pattern,
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    stage_aliases = {
        "initial": "initial",
        "first": "initial",
        "second": "second",
        "closing": "closing",
        "final": "final",
    }

    stages = []

    for stage_text in re.findall(
        r"\b(?:initial|first|second|closing|final)\b",
        match.group("stages"),
        flags=re.IGNORECASE,
    ):
        canonical_stage = stage_aliases[
            stage_text.lower()
        ]

        if canonical_stage not in stages:
            stages.append(canonical_stage)

    return stages or None
def extract_instrument_sets(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extract explicitly named surgical instrument sets.
    """
    normalized_transcript = transcript.strip().lower()

    matches = re.finditer(
        r"\b(?:the\s+)?"
        r"(?P<set_name>"
        r"[a-z][a-z-]*"
        r"(?:\s+[a-z][a-z-]*){0,2}"
        r"\s+instrument\s+sets?"
        r")\b",
        normalized_transcript,
        flags=re.IGNORECASE,
    )

    instrument_sets = []

    for match in matches:
        set_name = match.group(
            "set_name"
        ).strip()
        set_name = re.sub(
            r"^(?:(?:for|and|the)\s+)+",
            "",
            set_name,
            flags=re.IGNORECASE,
        )


        set_name = re.sub(
            r"\s+sets$",
            " set",
            set_name,
            flags=re.IGNORECASE,
        )

        if set_name not in instrument_sets:
            instrument_sets.append(
                set_name
            )

    return instrument_sets or None
def extract_irrigation_solution(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extract a known irrigation solution only when the clause
    explicitly documents irrigation or operative-field washing.
    """

    normalized = transcript.strip().lower()

    context_pattern = (
        r"\b(?:"
        r"irrigation\s+(?:was\s+)?"
        r"(?:performed|used|carried\s+out)\s+"
        r"(?:using|with)|"
        r"(?:operative|surgical)\s+field\s+"
        r"(?:was\s+)?"
        r"(?:irrigated|washed|rinsed|flushed)\s+"
        r"(?:using|with)"
        r")\b"
    )

    if not re.search(
        context_pattern,
        normalized,
        flags=re.IGNORECASE,
    ):
        return None

    solution_patterns = {
        "lactated ringer's solution": (
            r"\b(?:lactated\s+(?:d?ringer|d?ringers|"
            r"d?ringer's)(?:\s+solution)?|lr)\b"
        ),
        "normal saline": (
            r"\b(?:normal\s+saline|0\.9%\s+saline)\b"
        ),
        "sterile water": (
            r"\b(?:sterile\s+water|"
            r"water\s+for\s+irrigation)\b"
        ),
    }

    for canonical_value, pattern in solution_patterns.items():
        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):
            return [canonical_value]

    return None
def extract_structured_list(
    transcript: str,
    metadata: dict[str, str],
) -> Any | None:
    """
    Select the appropriate structured-list parser
    according to the predicted clinical field.
    """

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    structured_parsers = {
    "medication_details": (
        extract_medication_details
    ),
    "specimen_details": (
        extract_specimen_details
    ),
     "drain_details": (
            extract_drain_details
        ),
    "implant_details": (
            extract_implant_details
        ),
    "irrigation_solution": (
            extract_irrigation_solution
        ),
        "count_stage_details": (
    extract_count_stage_details
        ),
        "instrument_sets": (
            extract_instrument_sets
        ),
     "irrigation_solution": extract_irrigation_solution,   
}

    parser = structured_parsers.get(field_name)

    if parser is None:
        return None

    return parser(
        transcript,
        metadata,
    )
def extract_count_actions(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extracts corrective actions taken after an
    incorrect surgical count.
    """

    normalized_transcript = transcript.strip().lower()

    if not normalized_transcript:
        return None

    matched_actions = []

    action_patterns = {
        "head nurse informed": (
            r"\b(?:head|charge)\s+nurse\s+"
            r"(?:was\s+)?(?:informed|notified)\b"
        ),
        "x-ray taken": (
            r"\b(?:x[-\s]?ray|radiograph)\s+"
            r"(?:was\s+)?(?:taken|performed|obtained)\b"
        ),
        "incident report completed": (
            r"\bincident\s+report\s+"
            r"(?:was\s+)?(?:completed|filed|submitted)\b"
        ),
        "surgical field checked again": (
            r"\b(?:the\s+)?"
            r"(?:surgical|operative)?\s*field\s+"
            r"(?:was\s+)?"
            r"(?:checked|searched|inspected|examined)"
            r"(?:\s+again)?\b"
        ),
        "count repeated": (
            r"\b(?:"
            r"(?:count|counts)\s+"
            r"(?:(?:was|were)\s+)?"
            r"(?:repeated|rechecked|performed\s+again)|"
            r"(?:a\s+)?repeat\s+count\s+"
            r"(?:was\s+)?completed"
            r")\b"
        ),
        "wound reopened": (
            r"\b(?:the\s+)?wound\s+"
            r"(?:was\s+)?reopened\b"
        ),

        "missing sponge removed": (
            r"\b(?:the\s+)?missing\s+sponge\s+"
            r"(?:was\s+)?removed\b"
        ),
    }

    for normalized_action, pattern in (
        action_patterns.items()
    ):
        match = re.search(
            pattern,
            normalized_transcript,
            flags=re.IGNORECASE,
        )

        if match is not None:
            matched_actions.append(
                (
                    match.start(),
                    normalized_action,
                )
            )

    if not matched_actions:
        return None

    matched_actions.sort(
        key=lambda item: item[0]
    )
    return [
        action
        for _, action in matched_actions
    ]
def extract_event_list(
    transcript: str,
    metadata: dict[str, str],
) -> list[str] | None:
    """
    Extracts an unexpected event or an explicitly
    documented complication.

    The validation patterns are field-specific so that
    one event is not automatically accepted for every
    event-related field.
    """

    clinical_context = re.search(r"\b(?:during\s+(?:the\s+)?(?:surgery|operation|procedure)|"
                                 r"intraoperative|operative\s+field)\b", transcript, re.I)
    hypothetical_or_negative = re.search(r"\b(?:no|not|never|denies|without|"
                                          r"possible|potential|suspected|risk\s+of|"
                                          r"might|could|would|if)\b", transcript, re.I)
    if clinical_context and not hypothetical_or_negative:
        accidental_injury = re.search(r"\b(?:accidental|unintended|inadvertent)\s+"
                                      r"(?:[a-z-]+\s+){0,5}(?:burn|injury|damage|"
                                      r"perforation|laceration)\b.{0,50}"
                                      r"\b(?:occurred|was\s+identified|was\s+treated)\b",
                                      transcript, re.I)
        device_failure = re.search(r"\b(?:suction\s+(?:device|unit)|equipment|device|unit)"
                                   r"\s+(?:unexpectedly\s+)?(?:stopped\s+"
                                   r"(?:drawing\s+fluid|working|functioning)|failed|malfunctioned)\b",
                                   transcript, re.I)
        field = metadata.get("field")
        if accidental_injury and field in {"unexpected_event", "complications"}:
            return [transcript.strip()]
        if device_failure and field == "unexpected_event":
            return [transcript.strip()]

    event_text = transcript.strip()

    if not event_text:
        return None

    normalized_text = event_text.lower()

    field_name = metadata.get(
        "field",
        "",
    ).strip().lower()

    no_event_pattern = (
        r"\b(?:"
        r"no\s+unexpected\s+events?|"
        r"no\s+intraoperative\s+complications?|"
        r"no\s+complications?|"
        r"no\s+adverse\s+events?"
        r")\b"
    )

    if re.search(
        no_event_pattern,
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return None
        # A count problem becomes an unexpected event only when
    # the transcript also documents corrective action.
    #
    # This distinguishes:
    #   "A discrepancy was found during the final count."
    # from:
    #   "The count was incorrect and the field was checked."
    if field_name == "unexpected_event":
        count_issue_pattern = (
            r"\b(?:"
            r"(?:final\s+)?count\s+discrepancy|"
            r"(?:sponge|instrument|needle|final|surgical)\s+"
            r"count\s+(?:was\s+)?incorrect|"
            r"counts?\s+(?:was|were)\s+incorrect|"
            r"incorrect\s+(?:sponge\s+|instrument\s+|"
            r"needle\s+)?count|"
            r"discrepancy.{0,60}\b"
            r"(?:final\s+)?(?:surgical\s+)?count"
            r")\b"
        )

        count_issue_match = re.search(
            count_issue_pattern,
            normalized_text,
            flags=re.IGNORECASE,
        )

        if count_issue_match is not None:
            count_actions = extract_count_actions(
                transcript,
                {
                    "field": "count_actions",
                },
            )

            # A count discrepancy without documented
            # intervention remains only a count field.
            if not count_actions:
                return None

            return [event_text]

    field_patterns = {
    "unexpected_event": (
        r"\b(?:"
        r"unexpected|"
        r"unplanned|"
        r"(?:"
        r"(?:oxygen\s+saturation|blood\s+pressure|"
        r"heart\s+rate)\s+"
        r"(?:suddenly\s+)?"
        r"(?:decreased|dropped|increased|rose|fell)|"
        r"(?:sudden\s+)?"
        r"(?:decrease|drop|increase|rise|fall)\s+in\s+"
        r"(?:oxygen\s+saturation|blood\s+pressure|"
        r"heart\s+rate)"
        r")|"
        r"adverse\s+event|"
        r"complication|"
        r"discrepancy|"
        r"malfunction(?:ed|ing)?|"
        r"stopp(?:ed|ing)\s+working|"
        r"(?:failed|failure)\s+(?:during|before)\s+"
        r"(?:the\s+)?(?:operation|procedure|surgery)|"
        r"(?:equipment|device|unit)\s+(?:failed|failure)|"
        r"bleeding|"
        r"converted|"
        r"conversion|"
        r"aborted|"
        r"cancelled|"
        r"change(?:d)?\s+(?:in|to)\s+the\s+"
        r"(?:procedure|surgical\s+plan)"
        r")\b"
    ),

        "complications": (
            r"\b(?:"
            r"complications?|"
            r"complicated\s+by|"
            r"adverse\s+clinical\s+outcome|"
            r"allergic\s+reaction|"
            r"anaphylaxis|"
            r"anaphylactic\s+reaction|"
            r"adverse\s+(?:drug|medication)\s+reaction|"
            r"(?:unintended|accidental|inadvertent)\s+"
            r".{0,60}\b"
            r"(?:injury|damage|perforation|laceration)\b"
            r")\b"
        ),
    }

    selected_pattern = field_patterns.get(field_name)

    if selected_pattern is None:
        return None

    if not re.search(
        selected_pattern,
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return None

    return [event_text]

def extract_date(
    transcript: str,
    metadata: dict[str, str],
) -> str | None:
    """Extract a written month, day, and year as YYYY-MM-DD."""
    match = re.search(
        r"\b(?P<month>[A-Za-z]+)\s+"
        r"(?P<day>\d{1,2}),?\s+"
        r"(?P<year>\d{4})\b",
        transcript,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    text = (
        f"{match.group('month')} "
        f"{match.group('day')} "
        f"{match.group('year')}"
    )
    for date_format in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return None
def extract_semantic_value(
    transcript: str,
    metadata: dict[str, str],
) -> Any | None:
    """
    Selects a reusable value parser according to
    the field's configured parser strategy.
    """
    parser_strategy = metadata.get(
        "parser_strategy",
        "",
    ).strip().lower()

    parsers = {
    "measurement": extract_measurement,
    "boolean": extract_boolean,
    "person_name": extract_person_name,
    "categorical": extract_categorical,
    "categorical_list": extract_categorical_list,
    "identifier": extract_identifier,
    "allergy_list": extract_allergy_list,
    "time": extract_time,
    "event_text": extract_event_text,
    "event_list": extract_event_list,
    "count_action_list": extract_count_actions,
    "procedure_text": extract_procedure_text,
    "text": extract_text,
    "structured": extract_structured_list,
    "structured_list": extract_structured_list,
    "list": extract_narrative_list,
    "date": extract_date,
}

    parser = parsers.get(parser_strategy)

    if parser is None:
        return None

    return parser(transcript, metadata)


def convert_value(value: str) -> Any:
    """
    Converts an extracted string into a suitable
    Python data type.
    """
    normalized_value = value.strip().lower()

    if normalized_value in ["yes", "true", "confirmed"]:
        return True

    if normalized_value in ["no", "false", "not confirmed"]:
        return False

    if normalized_value.isdigit():
        return int(normalized_value)

    try:
        return float(normalized_value)
    except ValueError:
        return value.strip()


def convert_value_by_type(
    value: Any,
    data_type: str,
) -> Any:
    """
    Convert an extracted value according to its
    configured data type.
    """

    normalized_type = data_type.strip().lower()

    if normalized_type == "list":
        if isinstance(value, list):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "List values must be a list or string."
            )

        return [
            item.strip()
            for item in value.split("|")
            if item.strip()
        ]

    if not isinstance(value, str):
        raise TypeError(
            f"Cannot convert {type(value).__name__} "
            f"to {data_type}."
        )

    normalized_value = value.strip()

    if normalized_type == "int":
        return int(normalized_value)

    if normalized_type == "float":
        return float(normalized_value)

    if normalized_type == "bool":
        boolean_values = {
            "yes": True,
            "true": True,
            "confirmed": True,
            "no": False,
            "false": False,
            "not confirmed": False,
        }

        converted_value = boolean_values.get(
            normalized_value.lower()
        )

        if converted_value is None:
            raise ValueError(
                f"Cannot convert '{value}' to bool."
            )

        return converted_value

    if normalized_type == "str":
        return normalized_value

    raise ValueError(
        f"Unsupported data type: '{data_type}'."
    )

def parse_and_validate(
    transcript: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    """
    Extracts, converts, and validates a value for one
    semantically predicted field.

    A prediction is accepted only when:
    1. A configured parser exists.
    2. The parser extracts a value.
    3. The value can be converted to the configured type.
    """

    field_name = metadata.get(
        "field",
        "",
    ).strip()

    parser_strategy = metadata.get(
        "parser_strategy",
        "",
    ).strip()

    data_type = metadata.get(
        "data_type",
        "",
    ).strip()

    if not parser_strategy:
        return {
            "field": field_name,
            "accepted": False,
            "extracted_value": None,
            "converted_value": None,
            "reason": "No parser strategy is configured.",
        }

    extracted_value = extract_semantic_value(
        transcript=transcript,
        metadata=metadata,
    )

    if extracted_value is None:
        return {
            "field": field_name,
            "accepted": False,
            "extracted_value": None,
            "converted_value": None,
            "reason": (
                "The predicted field did not contain "
                "a valid extractable value."
            ),
        }

    try:
        converted_value = convert_value_by_type(
            value=extracted_value,
            data_type=data_type,
        )

    except (ValueError, TypeError) as error:
        return {
            "field": field_name,
            "accepted": False,
            "extracted_value": extracted_value,
            "converted_value": None,
            "reason": f"Value conversion failed: {error}",
        }

    return {
        "field": field_name,
        "accepted": True,
        "extracted_value": extracted_value,
        "converted_value": converted_value,
        "reason": "Valid value extracted and converted.",
    }

if __name__ == "__main__":
    metadata = {
        "parser_strategy": "event_text",
        "data_type": "str",
        "unit": "",
        "allowed_values": "",
    }

    transcript = (
        "An unexpected event occurred: equipment malfunction."
    )

    extracted_value = extract_semantic_value(
        transcript=transcript,
        metadata=metadata,
    )

    converted_value = (
        convert_value_by_type(
            value=extracted_value,
            data_type=metadata["data_type"],
        )
        if extracted_value is not None
        else None
    )

    print("Extracted value:", extracted_value)
    print("Converted value:", converted_value)
    print(
        "Converted type:",
        type(converted_value).__name__,
    )
