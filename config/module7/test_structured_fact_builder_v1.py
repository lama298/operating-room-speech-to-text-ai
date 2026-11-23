import json
from pathlib import Path

from structured_fact_builder_v1 import  StructuredFactBuilderV1
from llm_prompt_builder_v1 import StrictClinicalPromptBuilderV1


PROJECT_ROOT= Path(__file__).resolve().parents[2]
RECORD_PATH=(
    PROJECT_ROOT
    /"outputs"
    /"structured_case_v1.json"
)
def load_record(record_path:Path)->dict:
    if not record_path.exists():
        raise FileNotFoundError(f"Record not found: {record_path}")

    with record_path.open("r",encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    record = load_record(
        RECORD_PATH
    )

    builder = StructuredFactBuilderV1()

    facts = builder.build_facts(
        record
    )

    llm_input = builder.render_facts(
        facts
    )

    print("SOURCE RECORD")
    print(RECORD_PATH)

    print("\nFACT COUNT")
    print(len(facts))

    print("\nLLM INPUT")
    print("=" * 70)
    print(llm_input)
    prompt_builder = (
        StrictClinicalPromptBuilderV1()
    )

    messages = prompt_builder.build_messages(
        llm_input=llm_input,
        fact_count=len(facts),
    )

    print("\nLLAMA CHAT MESSAGES")
    print("=" * 70)

    for message in messages:
        print(
            f"\nROLE: {message['role'].upper()}"
        )

        print(message["content"])