from llm_sdk.llm_sdk import Small_LLM_Model
from trie import Trie, TrieNode

def ft_tokinize_function(model: Small_LLM_Model, functions: list[dict]) -> list[list[int]] | None:

    tokens_func: list[list[int]] = []
    for function in functions:
        tokens_func.append(model.encode(function["name"])[0].tolist())

    if not tokens_func:
        return None

    return tokens_func

