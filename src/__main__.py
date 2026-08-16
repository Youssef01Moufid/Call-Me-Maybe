import json
from trie import Trie
from constraint_decoder import (
    ft_tokinize_function,
    build_targets,
    generate_from_targets,
    get_number_candidate_pool,
    get_safe_string_mask,
    select_function_name,
)
from loader import load_function_definitions, load_prompt_items
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionCallResult
from cli import parse_args


def build_selection_prompt(user_prompt: str, function_names: list[str]) -> str:
    """Build a prompt instructing the model to pick a function."""
    names_list = ", ".join(function_names)
    system_message = (
        f"Pick the function that matches this request. "
        f"Available functions: {names_list}."
    )
    return (
        f"{system_message}\n"
        f"Request: {user_prompt}\n"
        f"Answer:"
    )

def build_extraction_prompt(user_prompt: str, function: dict) -> str:
    """Build a prompt instructing the model to extract arguments."""
    system_message = (
        f"Extract the argument values for function '{function['name']}' "
        f"from the user's request. "
        f"Return the values exactly as they appear in the request. "
        f"Do not execute the function or transform, calculate, reverse, "
        f"translate, or modify any value. "
    )

    return (
        f"{system_message}\n"
        f"Request: {user_prompt}\n"
        f"Answer:"
    )

def main() -> None:
    """Generate full function-call results (name + parameters) for all prompts."""
    args = parse_args()
    functions_path = args.functions_definition
    prompts_path = args.input
    output_path = args.output

    try:
        function_defs = load_function_definitions(functions_path)
    except ValueError as e:
        print(f"Error loading functions definition: {e}")
        return

    try:
        prompt_items = load_prompt_items(prompts_path)
    except ValueError as e:
        print(f"Error loading prompts: {e}")
        return

    functions = [f.model_dump() for f in function_defs]
    prompts = [p.model_dump() for p in prompt_items]

    llm_model = Small_LLM_Model()

    vocab_path = llm_model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    trie = Trie()
    trie.build_from_vocab(vocab)

    number_candidate_pool = get_number_candidate_pool(id_to_token)

    sample_logits = llm_model.get_logits_from_input_ids([0])
    vocab_size = len(sample_logits)
    safe_string_mask = get_safe_string_mask(id_to_token, vocab_size)

    tokens_func = ft_tokinize_function(llm_model, functions)
    function_names = [f["name"] for f in functions]
    functions_by_name = {f["name"]: f for f in functions}

    results = []
    for prompt_item in prompts:
        prompt = prompt_item["prompt"]
        try:
            selection_prompt = build_selection_prompt(prompt, function_names)
            selection_ids = llm_model.encode(selection_prompt)[0].tolist()
            name = select_function_name(llm_model, selection_ids, tokens_func, function_names)

            function = functions_by_name[name]
            targets = build_targets(function)

            extraction_prompt = build_extraction_prompt(prompt, function)
            param_ids = llm_model.encode(extraction_prompt)[0].tolist()
            prefix_len = len(param_ids)

            minus_token_id = next(tid for tid, tok in id_to_token.items() if tok == "-")

            param_ids = generate_from_targets(
                llm_model, trie, id_to_token, param_ids, targets,
                number_candidate_pool, safe_string_mask, minus_token_id, prompt
            )

            parameters_json = llm_model.decode(param_ids[prefix_len:])
            parameters = json.loads(parameters_json)

            results.append(FunctionCallResult(prompt=prompt, name=name, parameters=parameters))
        except Exception as e:
            print(f"Warning: failed to process prompt {prompt!r}: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


if __name__ == "__main__":
    main()
