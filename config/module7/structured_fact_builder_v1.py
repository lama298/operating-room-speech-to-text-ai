from typing import Any

class StructuredFactBuilderV1:

    EXCLUDED_ROOT_FIELDS = {
        "_id",
        "schema_version",
        "processing_metadata",
    }
    UNIT_SUFFIXES = {
        "_ml": "ml",
        "_mmhg": "mmHg",
        "_minutes": "minutes",
        "_seconds": "seconds",
        "_celsius": "°C",
    }
    @staticmethod #we do not need to create an object to test emptiness
    def is_empty_value(value:Any)->bool: #return either true or false
        #return true only when value contains no usable info
        if value is None:
            return True
        if isinstance(value,str): #checks whether each value belongs to specific class, yaane if value is str return true
               #if value is string gave true, this result decide whether to execute next line
            return not value.strip() #remove whitespaces
        if isinstance(value,
                      (list,tuple,set,dict)):
            return len(value)==0

        return False

    def flatten_record(
            self,
            record: dict[str,Any], #schema
            parent_path: str = "", #stores the path leading to current dict
    )-> dict[str,Any]:
        flattened: dict[str,Any]= {}
        for key,value in record.items():
            if parent_path:
                current_path= f"{parent_path}.{key}"
            else:
                current_path=key

            if self.is_empty_value(value):
                continue

            if isinstance(value,dict):
                nested_values=self.flatten_record(
                    record=value,
                    parent_path=current_path,
                )
                flattened.update(nested_values)

            else:
                flattened[current_path] = value

        return flattened
    @staticmethod
    def humanize_path(path: str)->str:
        path_parts= path.split(".")
        readable_parts=[
            part.replace("_"," ").strip().capitalize()
            for part in path_parts
        ]

        return " - ".join(readable_parts)

    @staticmethod
    def format_value(value:Any) ->str:
        if isinstance(value,bool):
            return "yes" if value else "no"

        if isinstance(value,list):
            formatted_items=[
                str(item).strip()
                for item in value
                if not StructuredFactBuilderV1.is_empty_value(item)
            ]
            return ", ".join(formatted_items)
        if isinstance(value,float):
            if value.is_integer():
                return str(int(value))
            return str(value)
        return str(value).strip()

    def separate_unit(
        self,
        path: str,
    ) -> tuple[str, str | None]:


        for suffix, unit in self.UNIT_SUFFIXES.items():
            if path.endswith(suffix):
                clean_path = path[
                    :-len(suffix)
                ]

                return clean_path, unit

        return path, None

    def build_facts(self,record:dict[str,Any],)->list[dict[str,Any]]:
        flattened_record=self.flatten_record(record)
        facts: list[dict[str,Any]]=[]

        sorted_paths=sorted(flattened_record.keys())
        for path in sorted_paths:
            root_field=path.split(".",1)[0]

            if root_field in self.EXCLUDED_ROOT_FIELDS:
               continue

            raw_value=flattened_record[path]
            formatted_value=self.format_value(raw_value)
            if self.is_empty_value(formatted_value):
                continue

            fact_number=len(facts)+1
            fact_id=f"F{fact_number:03d}"
            display_path, unit = self.separate_unit(
                 path
                )

            label = self.humanize_path(
                display_path
            )

            if unit:
                llm_value = (
                    f"{formatted_value} {unit}"
                )
            else:
                llm_value = formatted_value

            statement = (
                f"{label} = {llm_value}"
            )
            facts.append({
                    "fact_id": fact_id,
                    "schema_path": path,
                    "label": label,
                    "raw_value": raw_value,
                    "formatted_value": formatted_value,
                    "statement": statement,
                    "unit": unit,
                    "llm_value": llm_value,
                })
        return facts


    @staticmethod
    def render_facts(
        facts: list[dict[str, Any]],
    ) -> str:


        rendered_facts: list[str] = []

        for fact in facts:
            rendered_fact = (
                f"{fact['fact_id']}: "
                f"{fact['statement']}"
            )

            rendered_facts.append(
                rendered_fact
            )

        return "\n".join(
            rendered_facts
        )


if __name__=="__main__":
    builder= StructuredFactBuilderV1()

    test_record={
        "diagnosis_and_procedure":{
            "preoperative_diagnosis":("acute appendicitis"),
            "postoperative_diagnosis": None,
        },
         "anesthesia": {
            "types": ["general"],
        },
        "fluids_and_blood": {
            "estimated_blood_loss_ml": 50.0,
            "blood_transfusion_given": False,
            "blood_transfused_ml": 0,
        },
        "patient_outcome": {
            "condition": "stable",
            "other_destination": "   ",
        },

    }

    flattened_record=builder.flatten_record(
        test_record
    )

    print("Flattened record")
    print("="*60)

    for path,value in flattened_record.items():
        print(f"{path}={value!r}")

        print("\nREADABLE PATHS")
    print("=" * 60)

    for path in flattened_record:
        readable_label = builder.humanize_path(
            path
        )

        print(f"{path} -> {readable_label}")


        facts = builder.build_facts(
        test_record
    )

    print("\nSTRUCTURED FACTS")
    print("=" * 60)

    for fact in facts:
        print(
            f"{fact['fact_id']} | "
            f"{fact['schema_path']}"
        )

        print(
            f"  Label: {fact['label']}"
        )

        print(
            f"  Raw value: "
            f"{fact['raw_value']!r}"
        )

        print(
            f"  Formatted value: "
            f"{fact['formatted_value']!r}"
        )

        print(
            f"  Statement: "
            f"{fact['statement']}"
        )

    llm_input = builder.render_facts(
        facts
    )

    print("\nLLM FACT INPUT")
    print("=" * 60)
    print(llm_input)