from html import escape
from pathlib import Path
from typing import Any
from datetime import datetime

try:
    # Used when imported through config.module7.
    from .mongodb_storage import MongoDBStorage

except ImportError:
    # Allows this file to run directly.
    from mongodb_storage import MongoDBStorage


SECTION_LABELS = {
    "case_information": "Case Information",
    "patient_information": "Patient Information",
    "surgical_team": "Surgical Team",
    "diagnosis_and_procedure": "Diagnosis and Procedure",
    "anesthesia": "Anesthesia",
    "patient_position": "Patient Position",
    "irrigation": "Irrigation",
    "drains_tubes_and_packing": "Drains, Tubes, and Packing",
    "specimens_and_laboratory": "Specimens and Laboratory",
    "medications": "Medications",
    "implants": "Implants",
    "fluids_and_blood": "Fluids and Blood",
    "counts": "Counts",
    "equipment": "Equipment",
    "wound_classification": "Wound Classification",
    "complications": "Complications",
    "intraoperative_events": "Intraoperative Events",
    "patient_outcome": "Patient Outcome",
    "administrative": "Administrative",
}


FIELD_LABELS = {
    "preoperative_diagnosis": "Preoperative Diagnosis",
    "postoperative_diagnosis": "Postoperative Diagnosis",
    "procedure_name": "Procedure Name",
    "procedure_description": "Procedure Description",
    "types": "Anesthesia Type",
    "position": "Position",
    "prophylactic_antibiotic_given": (
        "Prophylactic Antibiotic Given"
    ),
    "collected": "Specimen Collected",
    "sent_to_external_lab": "Sent to External Laboratory",
    "estimated_blood_loss_ml": "Estimated Blood Loss",
    "total_fluids_ml": "Total Fluids",
    "blood_transfused_ml": "Blood Transfused",
    "final_count_correct": "Final Count Correct",
    "condition": "Patient Condition",
    "destination": "Destination",
    "unexpected_events": "Unexpected Events",
}


UNIT_BY_FIELD = {
    "estimated_blood_loss_ml": "ml",
    "total_fluids_ml": "ml",
    "blood_transfused_ml": "ml",
    "pressure_mmhg": "mmHg",
    "duration_minutes": "minutes",
}


ALERT_SECTIONS = {
    "complications",
    "intraoperative_events",
}


def has_value(value: Any) -> bool:
    """
    Return True when a schema value contains reportable
    information. False remains a valid reportable value.
    """

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return any(has_value(item) for item in value)

    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())

    return True


def make_readable_label(field_name: str) -> str:
    """Convert an internal field name to a report label."""

    configured_label = FIELD_LABELS.get(field_name)

    if configured_label:
        return configured_label

    return field_name.replace("_", " ").title()


def format_value(field_name: str, value: Any) -> str:
    """Convert a Python value into readable report text."""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, list):
        formatted_items = [
            format_value(field_name, item)
            for item in value
            if has_value(item)
        ]

        return ", ".join(formatted_items)

    if isinstance(value, float) and value.is_integer():
        displayed_value = str(int(value))
    else:
        displayed_value = str(value)

    unit = UNIT_BY_FIELD.get(field_name)

    if unit:
        displayed_value = f"{displayed_value} {unit}"

    return displayed_value


def collect_report_rows(
    section_data: dict[str, Any],
    parent_label: str = "",
) -> list[tuple[str, str]]:
    """Recursively collect populated fields from a section."""

    rows = []

    for field_name, value in section_data.items():
        if not has_value(value):
            continue

        field_label = make_readable_label(field_name)

        if parent_label:
            complete_label = f"{parent_label} — {field_label}"
        else:
            complete_label = field_label

        if isinstance(value, dict):
            rows.extend(
                collect_report_rows(
                    section_data=value,
                    parent_label=complete_label,
                )
            )
        else:
            rows.append(
                (
                    complete_label,
                    format_value(field_name, value),
                )
            )

    return rows


def _format_timestamp(value: Any) -> str:
    """Format an ISO timestamp for clinical report display."""

    if not isinstance(value, str) or not value.strip():
        return "Not available"

    normalized_value = value.strip().replace(
        "Z",
        "+00:00",
    )

    try:
        timestamp = datetime.fromisoformat(
            normalized_value
        )
    except ValueError:
        return value.replace("T", " ")

    formatted_time = timestamp.strftime(
        "%B %Y, %I:%M %p"
    ).lstrip("0")

    return f"{timestamp.day} {formatted_time}"


def generate_html_report(record: dict[str, Any]) -> str:
    """
    Generate a deterministic, responsive, print-ready
    clinical report from a Schema V2 MongoDB document.
    """

    record_id = str(record.get("_id", "Unknown"))

    processing_metadata = record.get("processing_metadata", {})

    if not isinstance(processing_metadata, dict):
        processing_metadata = {}

    generated_at = _format_timestamp(
        processing_metadata.get("session_finalized_at")
        or processing_metadata.get("processed_at")
    )
    section_html = []

    for section_name, section_data in record.items():
        if section_name in {
            "_id",
            "schema_version",
            "processing_metadata",
        }:
            continue

        if not has_value(section_data):
            continue

        section_label = SECTION_LABELS.get(
            section_name,
            make_readable_label(section_name),
        )

        if isinstance(section_data, dict):
            rows = collect_report_rows(section_data)
        else:
            rows = [
                (
                    section_label,
                    format_value(section_name, section_data),
                )
            ]

        if not rows:
            continue

        row_html = []

        for label, value in rows:
            row_html.append(
                '<div class="clinical-row">'
                f'<div class="field-label">{escape(label)}</div>'
                f'<div class="field-value">{escape(value)}</div>'
                "</div>"
            )

        section_number = len(section_html) + 1
        alert_class = (
            " alert-section"
            if section_name in ALERT_SECTIONS
            else ""
        )

        section_html.append(
            f'<section class="report-section{alert_class}">'
            '<div class="section-heading">'
            f'<span class="section-number">{section_number:02d}</span>'
            f'<h2>{escape(section_label)}</h2>'
            "</div>"
            f'<div class="section-body">{"".join(row_html)}</div>'
            "</section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>Clinical Report — {escape(record_id)}</title>

    <style>
        :root {{
            --navy: #123b5d;
            --blue: #176b9d;
            --blue-light: #eaf4fa;
            --ink: #172b3a;
            --muted: #607585;
            --line: #dce6ed;
            --page: #eef3f6;
            --card: #ffffff;
            --success: #17795b;
            --success-bg: #e8f7f1;
            --warning: #a34a2a;
            --warning-bg: #fff3ed;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: var(--page);
            color: var(--ink);
            font-family: "Segoe UI", Arial, sans-serif;
            line-height: 1.5;
        }}

        .toolbar {{
            position: sticky;
            top: 0;
            z-index: 10;
            display: flex;
            justify-content: flex-end;
            max-width: 1120px;
            margin: 0 auto;
            padding: 14px 20px 0;
        }}

        .print-button {{
            border: 0;
            border-radius: 8px;
            padding: 10px 17px;
            background: var(--navy);
            color: white;
            font-size: 14px;
            font-weight: 650;
            cursor: pointer;
            box-shadow: 0 5px 14px rgba(18, 59, 93, 0.18);
        }}

        .print-button:hover {{
            background: var(--blue);
        }}

        .report {{
            max-width: 1080px;
            margin: 14px auto 45px;
            background: var(--card);
            border: 1px solid #d9e4eb;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 18px 50px rgba(30, 59, 78, 0.12);
        }}

        .report-header {{
            position: relative;
            padding: 40px 44px 34px;
            background: linear-gradient(135deg, #103a5c, #1673a7);
            color: white;
        }}

        .report-header::after {{
            content: "";
            position: absolute;
            right: -55px;
            top: -85px;
            width: 240px;
            height: 240px;
            border: 35px solid rgba(255, 255, 255, 0.07);
            border-radius: 50%;
        }}

        .eyebrow {{
            margin: 0 0 8px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.8px;
            text-transform: uppercase;
            opacity: 0.78;
        }}

        h1 {{
            margin: 0;
            font-size: clamp(30px, 4vw, 43px);
            line-height: 1.14;
            letter-spacing: -0.7px;
        }}

        .header-subtitle {{
            max-width: 670px;
            margin: 12px 0 0;
            font-size: 15px;
            opacity: 0.86;
        }}

        .identity-strip {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1px;
            background: var(--line);
            border-bottom: 1px solid var(--line);
        }}

        .identity-item {{
            min-height: 88px;
            padding: 18px 22px;
            background: #f8fbfd;
        }}

        .identity-label {{
            display: block;
            margin-bottom: 5px;
            color: var(--muted);
            font-size: 11px;
            font-weight: 750;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}

        .identity-value {{
            color: var(--navy);
            font-size: 16px;
            font-weight: 700;
            overflow-wrap: anywhere;
        }}
        .sections-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
            padding: 24px 34px 34px;
        }}

        .report-section {{
            align-self: start;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 13px;
            background: white;
            break-inside: avoid;
        }}

        .section-heading {{
            display: flex;
            align-items: center;
            gap: 11px;
            padding: 14px 16px;
            background: var(--blue-light);
            border-bottom: 1px solid var(--line);
        }}

        .section-number {{
            display: grid;
            place-items: center;
            width: 30px;
            height: 30px;
            flex: 0 0 30px;
            border-radius: 8px;
            background: var(--blue);
            color: white;
            font-size: 11px;
            font-weight: 800;
        }}

        h2 {{
            margin: 0;
            color: var(--navy);
            font-size: 17px;
            line-height: 1.25;
        }}

        .clinical-row {{
            display: grid;
            grid-template-columns: minmax(120px, 42%) 1fr;
            gap: 15px;
            padding: 13px 16px;
            border-bottom: 1px solid #e8eef2;
        }}

        .clinical-row:last-child {{
            border-bottom: 0;
        }}

        .field-label {{
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
        }}

        .field-value {{
            color: var(--ink);
            font-size: 14px;
            font-weight: 620;
            overflow-wrap: anywhere;
        }}

        .alert-section {{
            border-color: #f0c9b9;
        }}

        .alert-section .section-heading {{
            background: var(--warning-bg);
            border-color: #f0c9b9;
        }}

        .alert-section .section-number {{
            background: var(--warning);
        }}

        .alert-section h2 {{
            color: #7f351d;
        }}

        .report-footer {{
            display: flex;
            justify-content: space-between;
            gap: 25px;
            padding: 20px 34px;
            border-top: 1px solid var(--line);
            background: #f7fafc;
            color: var(--muted);
            font-size: 11px;
        }}

        .report-footer strong {{
            color: var(--navy);
        }}

        @media (max-width: 780px) {{
            .identity-strip,
            .sections-grid {{
                grid-template-columns: 1fr;
            }}

            .report {{
                margin: 0;
                border: 0;
                border-radius: 0;
            }}

            .report-header {{
                padding: 32px 24px;
            }}

            .sections-grid {{
                padding-left: 20px;
                padding-right: 20px;
            }}

            .clinical-row {{
                grid-template-columns: 1fr;
                gap: 4px;
            }}

            .report-footer {{
                flex-direction: column;
            }}
        }}

        @page {{
            size: A4;
            margin: 13mm;
        }}

        @media print {{
            body {{
                background: white;
                print-color-adjust: exact;
                -webkit-print-color-adjust: exact;
            }}

            .toolbar {{
                display: none;
            }}

            .report {{
                max-width: none;
                margin: 0;
                border: 0;
                border-radius: 0;
                box-shadow: none;
            }}

            .report-header {{
                padding: 24px 28px;
            }}

            .identity-item {{
                min-height: 70px;
                padding: 13px 15px;
            }}

            .sections-grid {{
                gap: 12px;
                padding: 18px 24px 24px;
            }}

            .section-heading {{
                padding: 10px 12px;
            }}

            .clinical-row {{
                padding: 9px 12px;
            }}
            .review-section {{
                margin: 0 34px 30px;
                padding: 18px 20px;
                border: 1px solid var(--line);
                border-radius: 12px;
                background: #f8fbfd;
            }}

            .review-section h2 {{
                margin-bottom: 12px;
            }}

            .review-section p {{
                margin: 8px 0;
                font-size: 14px;
            }}

            .report-footer {{
                padding: 14px 24px;
            }}
        }}
    </style>
</head>

<body>
    <div class="toolbar">
        <button class="print-button" onclick="window.print()">
            Print / Save as PDF
        </button>
    </div>

    <main class="report">
        <header class="report-header">
            <p class="eyebrow">Operating Room Documentation</p>

            <h1>Operating Room Documentation Summary</h1>

            <p class="header-subtitle">
                Draft – requires clinician review
            </p>
        </header>

        <section class="identity-strip">
            <div class="identity-item">
                <span class="identity-label">Case ID</span>
                <span class="identity-value">{escape(record_id)}</span>
            </div>

            <div class="identity-item">
                <span class="identity-label">Report Generated</span>
                <span class="identity-value">{escape(generated_at)}</span>
            </div>
        </section>

        <div class="sections-grid">
            {''.join(section_html)}
        </div>
        <section class="review-section">
            <h2>Clinical Review</h2>
            <p>
                Reviewed by: ______________________________________
            </p>
            <p>
                Date and time: ____________________________________
            </p>
        </section>
        <footer class="report-footer">
        <div>
            Automatically generated from validated clinical documentation.
        </div>
</footer>

    </main>
</body>
</html>
"""


if __name__ == "__main__":
    storage = MongoDBStorage()

    try:
        record = storage.get_record("STD_001")

        if record is None:
            raise ValueError(
                "MongoDB record 'STD_001' was not found."
            )

        report_html = generate_html_report(record)

        project_root = Path(__file__).resolve().parents[2]

        output_path = (
            project_root
            / "outputs"
            / "end_to_end_test"
            / "clinical_report_STD_001.html"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_html, encoding="utf-8")

        update_result = storage.mark_report_generated("STD_001")

        if update_result["matched_count"] != 1:
            raise RuntimeError(
                "The report was created, but MongoDB record "
                "'STD_001' could not be updated."
            )

        print("Report generated successfully.")
        print("Output:", output_path)
        print("MongoDB report status updated:", update_result)

    finally:
        storage.close()
