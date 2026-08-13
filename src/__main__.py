import json
from pathlib import Path

from trie import Trie
from constraint_decoder import (
    ft_tokinize_function,
    build_targets,
    generate_from_targets,
    get_digit_tokens,
    get_safe_string_tokens,
    select_function_name,
)
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionCallResult


def build_selection_prompt(user_prompt: str, function_names: list[str]) -> str:
    """Build a chat-formatted prompt instructing the model to pick a function."""
    names_list = ", ".join(function_names)
    system_message = (
        f"Pick the function that matches the user's request. "
        f"Available functions: {names_list}."
    )
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_extraction_prompt(user_prompt: str, function: dict) -> str:
    """Build a chat-formatted prompt instructing the model to extract arguments."""
    system_message = (
        f"Extract the values for function '{function['name']}' "
        f"({function['description']}) from the user's message."
    )
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> None:
    """Generate full function-call results (name + parameters) for all prompts."""
    llm_model = Small_LLM_Model()

    functions_path = Path("data/input/functions_definition.json")
    prompts_path = Path("data/input/function_calling_tests.json")
    output_path = Path("data/output/function_calling_results.json")

    with open(functions_path, "r", encoding="utf-8") as f:
        functions = json.load(f)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    vocab_path = llm_model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    trie = Trie()
    trie.build_from_vocab(vocab)

    digit_ids = get_digit_tokens(id_to_token)
    safe_string_ids = get_safe_string_tokens(id_to_token)

    tokens_func = ft_tokinize_function(llm_model, functions)
    function_names = [f["name"] for f in functions]
    functions_by_name = {f["name"]: f for f in functions}

    results = []
    for prompt_item in prompts:
        prompt = prompt_item["prompt"]

        # Step 1: select the function
        selection_prompt = build_selection_prompt(prompt, function_names)
        selection_ids = llm_model.encode(selection_prompt)[0].tolist()
        name = select_function_name(llm_model, selection_ids, tokens_func, function_names)

        # Step 2: generate the parameters for that function
        function = functions_by_name[name]
        targets = build_targets(function)

        extraction_prompt = build_extraction_prompt(prompt, function)
        param_ids = llm_model.encode(extraction_prompt)[0].tolist()
        prefix_len = len(param_ids)

        param_ids = generate_from_targets(
            llm_model, trie, id_to_token, param_ids, targets, digit_ids, safe_string_ids
        )

        parameters_json = llm_model.decode(param_ids[prefix_len:])
        parameters = json.loads(parameters_json)

        results.append(FunctionCallResult(prompt=prompt, name=name, parameters=parameters))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()