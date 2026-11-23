from config.module7.value_parser import parse_and_validate


def check(transcript: str, metadata: dict, expected):
    result = parse_and_validate(transcript=transcript, metadata=metadata)
    assert result["accepted"] is (expected is not None), result
    if expected is not None:
        assert result["converted_value"] == expected, result


ACTIVE_PROCEDURE = (
    "Dr. Hala Yunis performed an open reduction and "
    "internal fixation of the left ankle."
)

check(
    ACTIVE_PROCEDURE,
    {"field": "surgeon_1", "parser_strategy": "person_name", "data_type": "str", "keywords": "primary surgeon,surgeon"},
    "Hala Yunis",
)
check(ACTIVE_PROCEDURE, {"field": "procedure_name", "parser_strategy": "text", "data_type": "str"}, "open reduction and internal fixation")
check(ACTIVE_PROCEDURE, {"field": "procedure_site", "parser_strategy": "text", "data_type": "str"}, "ankle")
check(ACTIVE_PROCEDURE, {"field": "laterality", "parser_strategy": "categorical", "data_type": "str", "allowed_values": "left|right|bilateral"}, "left")
check(
    "The patient remained intubated before transfer to the cardiac observation unit.",
    {"field": "other_destination", "parser_strategy": "text", "data_type": "str"},
    "cardiac observation unit",
)
check(
    "Dr. Hala Yunis discussed the procedure.",
    {"field": "surgeon_1", "parser_strategy": "person_name", "data_type": "str", "keywords": "primary surgeon,surgeon"},
    None,
)

print("Natural-language regression checks passed.")