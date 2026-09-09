import json
from json_schema_to_pydantic import create_model  # type: ignore


def load_functions(filepath: str) -> list[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dynamic_validators(functions: list[dict]) -> dict:
    return {
        func["name"]: create_model(func.get("parameters", {}))
        for func in functions
    }
