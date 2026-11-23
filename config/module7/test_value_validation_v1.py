import csv
from pathlib import Path

try:
    from .value_parser import parse_and_validate
except ImportError:
    from value_parser import parse_and_validate


def load_field_metadata():
    """
    Load field metadata and index it by field name.
    """

    metadata_path = (
        Path(__file__).parent
        / "field_sources_v2.csv"
    )

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


if __name__ == "__main__":
    field_metadata = load_field_metadata()

    test_cases = [
        {
            "field": "surgery_end_time",
            "transcript": (
                "The operation ended at 12:20."
            ),
            "expected_accepted": True,
            "expected_value": "12:20",
        },
        {
            "field": "surgery_end_time",
            "transcript": (
                "The procedure was completed successfully."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "specimen_collected",
            "transcript": (
                "The appendix specimen was collected."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "specimen_sent_to_external_lab",
            "transcript": (
                "The appendix specimen was sent to pathology."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "antibiotic_prophylaxis_given",
            "transcript": (
                "One gram of cefazolin was administered "
                "before incision."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "estimated_blood_loss_ml",
            "transcript": (
                "Estimated blood loss was 50 milliliters."
            ),
            "expected_accepted": True,
            "expected_value": 50.0,
        },
        {
            "field": "patient_status",
            "transcript": (
                "The patient remained stable."
            ),
            "expected_accepted": True,
            "expected_value": "stable",
        },
        {
            "field": "procedure_name",
            "transcript": (
                "The procedure is laparoscopic appendectomy."
            ),
            "expected_accepted": True,
            "expected_value": (
                "laparoscopic appendectomy"
            ),
        },
                {
            "field": "anesthesia_type",
            "transcript": (
                "General anesthesia was administered."
            ),
            "expected_accepted": True,
            "expected_value": ["general"],
        },
        {
            "field": "counts_correct",
            "transcript": (
                "The final instrument, sponge, and "
                "needle counts were correct."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "counts_correct",
            "transcript": (
                "The final counts were not correct."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "preoperative_diagnosis",
            "transcript": (
                "The preoperative diagnosis is "
                "acute appendicitis."
            ),
            "expected_accepted": True,
            "expected_value": "acute appendicitis",
        },
        {
            "field": "patient_name",
            "transcript": (
                "The patient is Maya Haddad and the medical "
                "record number is P2047."
            ),
            "expected_accepted": True,
            "expected_value": "Maya Haddad",
        },
        {
            "field": "surgeon_2",
            "transcript": (
                "The patient is Maya Haddad and the medical "
                "record number is P2047."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "assistant_1",
            "transcript": (
                "The patient is Maya Haddad and the medical "
                "record number is P2047."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "surgeon_1",
            "transcript": (
                "Doctor Karim Nassar is the primary surgeon, "
                "and Doctor Rania Salem is the anesthesiologist."
            ),
            "expected_accepted": True,
            "expected_value": "Karim Nassar",
        },
        {
            "field": "anesthesiologist",
            "transcript": (
                "Doctor Karim Nassar is the primary surgeon, "
                "and Doctor Rania Salem is the anesthesiologist."
            ),
            "expected_accepted": True,
            "expected_value": "Rania Salem",
        },
        {
            "field": "assistant_1",
            "transcript": (
                "Doctor Karim Nassar is the primary surgeon, "
                "and Doctor Rania Salem is the anesthesiologist."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "or_room_number",
            "transcript": (
                "The operation is taking place in operating "
                "room 4."
            ),
            "expected_accepted": True,
            "expected_value": "4",
        },
        {
            "field": "operation_code",
            "transcript": (
                "The operation is taking place in operating "
                "room 4."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "patient_id",
            "transcript": (
                "The patient is Maya Haddad and the medical "
                "record number is P2047."
            ),
            "expected_accepted": True,
            "expected_value": "P2047",
        },
        {
            "field": "case_id",
            "transcript": (
                "The patient is Maya Haddad and the medical "
                "record number is P2047."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "incision_time",
            "transcript": (
                "The skin incision was made at 9.35 "
                "in the morning."
            ),
            "expected_accepted": True,
            "expected_value": "09:35",
        },
        {
            "field": "incision_time",
            "transcript": (
                "The incision was made at 9:35."
            ),
            "expected_accepted": True,
            "expected_value": "09:35",
        },
        {
            "field": "incision_time",
            "transcript": (
                "The incision was made at 9.75."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "known_allergies",
            "transcript": (
                "The patient has a documented allergy "
                "to penicillin."
            ),
            "expected_accepted": True,
            "expected_value": ["penicillin"],
        },
        {
            "field": "known_allergies",
            "transcript": "Penicillin was administered.",
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "anesthesia_type",
            "transcript": (
                "Spinal anesthesia was administered for the procedure"
            ),
            "expected_accepted": True,
            "expected_value": ["spinal"],
        },
                {
            "field": "transfer_destination",
            "transcript": (
                "The patient was transferred to the "
                "intensive care unit."
            ),
            "expected_accepted": True,
            "expected_value": "icu",
        },
                {
            "field": "count_discrepancy",
            "transcript": (
                "A discrepancy was found during the "
                "final surgical count."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
                {
            "field": "medication_details",
            "transcript": (
                "2 grams of cefazolin were given as "
                "prophylactic antibiotics before incision."
            ),
            "expected_accepted": True,
            "expected_value": [
                {
                    "name": "cefazolin",
                    "dose": "2 grams",
                }
            ],
        },
                {
            "field": "total_fluids_ml",
            "transcript": (
                "The patient received 1200 milliliters "
                "of fluids."
            ),
            "expected_accepted": True,
            "expected_value": 1200.0,
        },
        {
            "field": "estimated_blood_loss_ml",
            "transcript": (
                "The patient received 1200 milliliters "
                "of fluids."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "laboratory_requests",
            "transcript": (
                "A tissue specimen was collected and "
                "sent for histopathology."
            ),
            "expected_accepted": True,
            "expected_value": ["histopathology"],
        },
        {
            "field": "specimen_details",
            "transcript": (
                "A tissue specimen from the left ovary "
                "was collected."
            ),
            "expected_accepted": True,
            "expected_value": ["left ovary tissue"],
        },
                {
            "field": "drain_inserted",
            "transcript": (
                "A Jackson Pratt drain was inserted "
                "and left in place."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "drain_inserted",
            "transcript": "No drain was inserted.",
            "expected_accepted": True,
            "expected_value": False,
        },
                {
            "field": "implant_used",
            "transcript": (
                "A titanium plate with serial number "
                "TP842 was implanted."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "implant_used",
            "transcript": "No implant was used.",
            "expected_accepted": True,
            "expected_value": False,
        },
                {
            "field": "tourniquet_used",
            "transcript": (
                "The tourniquet was used at 250 mm of "
                "mercury for 40 minutes."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "tourniquet_pressure_mmhg",
            "transcript": (
                "The tourniquet was used at 250 mm of "
                "mercury for 40 minutes."
            ),
            "expected_accepted": True,
            "expected_value": 250.0,
        },
        {
            "field": "tourniquet_duration_minutes",
            "transcript": (
                "The tourniquet was used at 250 mm of "
                "mercury for 40 minutes."
            ),
            "expected_accepted": True,
            "expected_value": 40.0,
        },
                {
            "field": "counts_correct",
            "transcript": (
                "The final instrument sponge and needle "
                "counts were correct with no discrepancy "
                "identified."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "count_discrepancy",
            "transcript": (
                "The final instrument sponge and needle "
                "counts were correct with no discrepancy "
                "identified."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "count_discrepancy",
            "transcript": "The sponge count was incorrect.",
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "count_discrepancy",
            "transcript": (
                "A specimen label discrepancy was identified and "
                "corrected before the specimen left the room."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "drain_details",
            "transcript": (
                "A Jackson Pratt drain was inserted "
                "and left in place."
            ),
            "expected_accepted": True,
            "expected_value": [
                "Jackson-Pratt drain"
            ],
        },
        {
            "field": "drain_details",
            "transcript": (
                "A drain was inserted."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "implant_details",
            "transcript": (
                "A titanium plate with serial number "
                "TP842 was implanted."
            ),
            "expected_accepted": True,
            "expected_value": [
                {
                    "type": "titanium plate",
                    "serial_number": "TP842",
                }
            ],
        },
        {
            "field": "implant_details",
            "transcript": (
                "A titanium plate was implanted."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "procedure_description",
            "transcript": (
                "Unexpected bleeding required conversion "
                "from the laparoscopic procedure to an "
                "open procedure."
            ),
            "expected_accepted": True,
            "expected_value": (
                "converted from laparoscopic to "
                "open procedure"
            ),
        },
        {
            "field": "procedure_description",
            "transcript": (
                "The laparoscopic procedure was converted "
                "to an open procedure because of difficult "
                "exposure."
            ),
            "expected_accepted": True,
            "expected_value": (
                "converted from laparoscopic to "
                "open procedure"
            ),
        },
        {
            "field": "procedure_description",
            "transcript": (
                "Conversion from laparoscopic to open "
                "may be considered."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "complications",
            "transcript": (
                "An unexpected allergic reaction occurred "
                "after the antibiotic."
            ),
            "expected_accepted": True,
            "expected_value": [
                "An unexpected allergic reaction occurred "
                "after the antibiotic."
            ],
        },
        {
            "field": "complications",
            "transcript": (
                "The patient has a documented allergy "
                "to penicillin."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "complications",
            "transcript": (
                "No intraoperative complications occurred."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "patient_in_time",
            "transcript": (
                "The patient entered the operating room "
                "at 06:10."
            ),
            "expected_accepted": True,
            "expected_value": "06:10",
        },
        {
            "field": "anesthesia_induction_time",
            "transcript": (
                "Anesthesia induction began at 06:25."
            ),
            "expected_accepted": True,
            "expected_value": "06:25",
        },
        {
            "field": "incision_time",
            "transcript": (
                "The first skin incision was performed "
                "at 06:40."
            ),
            "expected_accepted": True,
            "expected_value": "06:40",
        },
        {
            "field": "surgery_end_time",
            "transcript": (
                "The operation concluded at 09:15."
            ),
            "expected_accepted": True,
            "expected_value": "09:15",
        },
        {
            "field": "dressing_time",
            "transcript": (
                "The surgical dressing was applied "
                "at 09:28."
            ),
            "expected_accepted": True,
            "expected_value": "09:28",
        },
        {
            "field": "patient_out_time",
            "transcript": (
                "The patient departed the operating "
                "suite at 09:40."
            ),
            "expected_accepted": True,
            "expected_value": "09:40",
        },
        {
            "field": "or_room_number",
            "transcript": (
                "The patient entered the operating "
                "room at 07:15."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "or_room_number",
            "transcript": (
                "The operation is taking place in "
                "operating room 5A."
            ),
            "expected_accepted": True,
            "expected_value": "5A",
        },
                {
            "field": "case_id",
            "transcript": (
                "Surgical case Q912 is scheduled "
                "for tomorrow."
            ),
            "expected_accepted": True,
            "expected_value": "Q912",
        },
        {
            "field": "case_id",
            "transcript": (
                "Operative case Z305 will begin "
                "this morning."
            ),
            "expected_accepted": True,
            "expected_value": "Z305",
        },
        {
            "field": "case_id",
            "transcript": (
                "The team discussed another "
                "surgical case."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "case_id",
            "transcript": (
                "The surgery is scheduled for "
                "tomorrow."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "allergy_status_confirmed",
            "transcript": (
                "Allergy verification was completed."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "allergy_status_confirmed",
            "transcript": (
                "Allergy verification is pending."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "allergy_status_confirmed",
            "transcript": (
                "Allergy verification was not "
                "completed."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "known_allergies",
            "transcript": (
                "She has known allergies to iodine "
                "and adhesive."
            ),
            "expected_accepted": True,
            "expected_value": [
                "iodine",
                "adhesive",
            ],
        },
        {
            "field": "known_allergies",
            "transcript": (
                "She was evaluated for a possible "
                "allergy."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
                {
            "field": "isolation_required",
            "transcript": (
                "Airborne isolation is required."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "isolation_required",
            "transcript": (
                "Isolation is not required."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "isolation_required",
            "transcript": (
                "The team discussed isolation "
                "procedures."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "fall_risk",
            "transcript": (
                "The assessment identified an "
                "elevated risk of falling."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "fall_risk",
            "transcript": (
                "Fall risk is low."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "fall_risk",
            "transcript": (
                "The patient nearly fell."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "relief_scrub_nurse",
            "transcript": (
                "The scrub nurse is Hala Yunis, "
                "relieved by Meera Khoury."
            ),
            "expected_accepted": True,
            "expected_value": "Meera Khoury",
        },
        {
            "field": "relief_circulating_nurse",
            "transcript": (
                "The circulating nurse was Rawan Faris "
                "and was relieved by Janna Karam."
            ),
            "expected_accepted": True,
            "expected_value": "Janna Karam",
        },
        {
            "field": "relief_scrub_nurse",
            "transcript": (
                "The scrub nurse is Hala Yunis."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "relief_scrub_nurse",
            "transcript": (
                "The circulating nurse is Rawan Faris, "
                "relieved by Janna Karam."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "procedure_name",
            "transcript": (
                "The planned operation is open reduction "
                "and internal fixation of the right ankle."
            ),
            "expected_accepted": True,
            "expected_value": (
                "open reduction and internal fixation"
            ),
        },
        {
            "field": "procedure_site",
            "transcript": (
                "The planned operation is open reduction "
                "and internal fixation of the right ankle."
            ),
            "expected_accepted": True,
            "expected_value": "ankle",
        },
        {
            "field": "procedure_name",
            "transcript": (
                "The team discussed a planned operation."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "procedure_name",
            "transcript": (
                "The surgical procedure was completed by "
                "removing the lesion and closing the wound."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "operative_findings",
            "transcript": (
                "Operative inspection revealed healthy "
                "margins and no vascular injury."
            ),
            "expected_accepted": True,
            "expected_value": [
                "healthy margins",
                "no vascular injury",
            ],
        },
        {
            "field": "other_position",
            "transcript": (
                "The operative position was a modified "
                "beach chair configuration."
            ),
            "expected_accepted": True,
            "expected_value": (
                "modified beach chair configuration"
            ),
        },
        {
            "field": "other_position",
            "transcript": (
                "The patient was positioned in a reverse "
                "jackknife configuration."
            ),
            "expected_accepted": True,
            "expected_value": (
                "reverse jackknife configuration"
            ),
        },
        {
            "field": "other_position",
            "transcript": (
                "The team discussed the chair configuration."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "irrigation_used",
            "transcript": (
                "Irrigation was performed using normal saline."
            ),
            "expected_accepted": True,
            "expected_value": True,
        },
        {
            "field": "irrigation_used",
            "transcript": (
                "Irrigation was not required."
            ),
            "expected_accepted": True,
            "expected_value": False,
        },
        {
            "field": "irrigation_used",
            "transcript": (
                "The team discussed the irrigation equipment."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
        {
            "field": "irrigation_solution",
            "transcript": (
                "Irrigation was performed using 2 liters "
                "of lactated ranger solution."
            ),
            "expected_accepted": True,
            "expected_value": [
                "lactated Ringer's solution",
            ],
        },
        {
            "field": "irrigation_solution",
            "transcript": (
                "The surgical field was irrigated with "
                "normal saline."
            ),
            "expected_accepted": True,
            "expected_value": [
                "normal saline",
            ],
        },
        {
            "field": "irrigation_solution",
            "transcript": (
                "The patient received intravenous fluids."
            ),
            "expected_accepted": False,
            "expected_value": None,
        },
    ]

    passed_tests = 0

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        field_name = test_case["field"]

        metadata = field_metadata[field_name]

        result = parse_and_validate(
            transcript=test_case["transcript"],
            metadata=metadata,
        )

        passed = (
            result["accepted"]
            == test_case["expected_accepted"]
            and result["converted_value"]
            == test_case["expected_value"]
        )

        if passed:
            passed_tests += 1

        print("\n" + "=" * 60)
        print(f"TEST {index}: {'PASS' if passed else 'FAIL'}")
        print("Field:", field_name)
        print("Transcript:", test_case["transcript"])
        print(
            "Expected accepted:",
            test_case["expected_accepted"],
        )
        print(
            "Actual accepted:",
            result["accepted"],
        )
        print(
            "Expected value:",
            test_case["expected_value"],
        )
        print(
            "Extracted value:",
            result["extracted_value"],
        )
        print(
            "Converted value:",
            result["converted_value"],
        )
        print("Reason:", result["reason"])

    print("\n" + "=" * 60)
    print(
        f"FINAL RESULT: "
        f"{passed_tests}/{len(test_cases)} tests passed"
    )
