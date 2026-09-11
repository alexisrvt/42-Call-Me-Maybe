import json
from typing import Any, cast
from json_schema_to_pydantic import create_model


def load_functions(filepath: str) -> list[dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return cast(list[dict[str, Any]], json.load(f))


def create_dynamic_validators(
        functions: list[dict[str, Any]]) -> dict[str, Any]:
    validators: dict[str, Any] = {}
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
        validators[str(func["name"])] = create_model(schema)
    return validators
