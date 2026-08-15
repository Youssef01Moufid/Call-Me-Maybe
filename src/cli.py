import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the function calling tool.

    Returns:
        Parsed arguments with functions_definition, input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Convert natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to write the output JSON file.",
    )
    return parser.parse_args()
