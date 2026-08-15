import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from models import FunctionDefinition, PromptItem


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        A list of validated FunctionDefinition objects.

    Raises:
        ValueError: If the file is missing, invalid JSON, or doesn't match the schema.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    adapter = TypeAdapter(list[FunctionDefinition])
    try:
        functions = adapter.validate_python(raw_data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed for {path}: {e}")

    return functions


def load_prompt_items(path: Path) -> list[PromptItem]:
    """Load prompt items from a JSON file.

    Args:
        path: Path to the prompt_items.json file.

    Returns:
        A list of validated PromptItem objects.

    Raises:
        ValueError: If the file is missing, invalid JSON, or doesn't match the schema.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    adapter = TypeAdapter(list[PromptItem])
    try:
        prompt_items = adapter.validate_python(raw_data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed for {path}: {e}")

    return prompt_items
