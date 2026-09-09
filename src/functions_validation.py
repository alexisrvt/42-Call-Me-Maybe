import json
from json_schema_to_pydantic import create_model  # type: ignore


def load_functions(filepath: str) -> list[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dynamic_validators(functions: list[dict]) -> dict:
    validators = {}
    for func in functions:
        params = func.get("parameters", {})
        if "properties" not in params:
            schema = {
                "type": "object",
                "properties": params,
                "required": list(params.keys())
            }
        else:
            schema = params
        validators[func["name"]] = create_model(schema)
    return validators
