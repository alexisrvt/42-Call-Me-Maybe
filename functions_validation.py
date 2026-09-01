import json
from typing import Dict, Any
from pydantic import BaseModel, create_model


class ParameterDef(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


def load_functions(filepath: str) -> list[FunctionDef]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [FunctionDef(**func) for func in data]


def create_dynamic_validators(
        functions: list[FunctionDef]) -> dict[str, type[BaseModel]]:
    type_mapping = {
        "number": (float, ...),
        "string": (str, ...),
        "boolean": (bool, ...)
    }

    validators = {}

    for func in functions:
        fields: dict[str, Any] = {}
        for param_name, param_def in func.parameters.items():
            python_type = type_mapping.get(param_def.type, (str, ...))
            fields[param_name] = python_type
        DynamicModel = create_model(f"Validator_{func.name}", **fields)
        validators[func.name] = DynamicModel
    return validators
