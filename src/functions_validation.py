import json
from typing import Any, cast
from json_schema_to_pydantic import create_model


def load_functions(filepath: str) -> list[dict[str, Any]]:
    """
    Loads function definitions from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing function
        definitions.

    Returns:
        list[dict[str, Any]]: A list of dictionaries where each dictionary
        represents a function schema.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return cast(list[dict[str, Any]], json.load(f))


def create_dynamic_validators(
        functions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Creates dynamic Pydantic models for validation based on function schemas.

    Automatically normalizes legacy schemas (without a 'properties' key) into
    valid JSON Schemas before generating the Pydantic models.

    Args:
        functions (list[dict[str, Any]]): A list of parsed function schemas.

    Returns:
        dict[str, Any]: A dictionary mapping function names (str) to their
        dynamically created Pydantic models (type).
    """
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
