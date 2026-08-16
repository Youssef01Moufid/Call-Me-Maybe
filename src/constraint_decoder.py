from llm_sdk.llm_sdk import Small_LLM_Model
from trie import Trie
import numpy as np


# ----------GENERATE FUNCTION NAME----------------

def ft_tokinize_function(model: Small_LLM_Model, functions: list[dict]) -> list[list[int]] | None:

    tokens_func: list[list[int]] = []
    for function in functions:
        tokens_func.append(model.encode(function["name"])[0].tolist())

    if not tokens_func:
        return None

    return tokens_func


def select_function_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    tokens_func: list[list[int]],
    function_names: list[str],
) -> str:
    """Force the model to select one of the known function names.

    Args:
        model: The LLM wrapper.
        input_ids: Token IDs for the prompt (not mutated).
        tokens_func: One token-ID sequence per candidate function name.
        function_names: The function names, in the same order as tokens_func.

    Returns:
        The selected function name.
    """
    working_ids = list(input_ids)
    remaining_indices = list(range(len(tokens_func)))
    position = 0

    while True:
        allowed_next = list({
            tokens_func[i][position]
            for i in remaining_indices
            if len(tokens_func[i]) > position
        })

        if not allowed_next:
            raise ValueError("No candidate function name matched generation.")

        logits = model.get_logits_from_input_ids(working_ids)
        best_token_id = max(allowed_next, key=lambda token_id: logits[token_id])

        working_ids.append(best_token_id)

        remaining_indices = [
            i for i in remaining_indices
            if len(tokens_func[i]) > position and tokens_func[i][position] == best_token_id
        ]

        position += 1

        if len(remaining_indices) == 1 and len(tokens_func[remaining_indices[0]]) == position:
            return function_names[remaining_indices[0]]


# ----------GENERATE PARAMETERS----------------

def find_safe_candidates(trie: Trie, id_to_token: dict[int, str], remaining: str) -> list[int]:
    """Find token IDs whose string is a prefix of `remaining` (doesn't overshoot).

    Args:
        trie: The vocabulary trie.
        id_to_token: Reverse mapping from token ID to token string.
        remaining: The exact text still needed.

    Returns:
        Token IDs safe to generate next without producing extra text.
    """
    first_char = remaining[0]
    broad_candidates = trie.search_prefix(first_char)

    safe = []
    for token_id in broad_candidates:
        token_str = id_to_token[token_id]
        if remaining.startswith(token_str):
            safe.append(token_id)
    return safe


def generate_literal(
    model: Small_LLM_Model,
    trie: Trie,
    id_to_token: dict[int, str],
    input_ids: list[int],
    target: str,
) -> list[int]:
    """Force-generate tokens that spell out exactly `target`.

    Args:
        model: The LLM wrapper.
        trie: The vocabulary trie.
        id_to_token: Reverse mapping from token ID to token string.
        input_ids: Token IDs generated so far (will be extended).
        target: The exact literal string that must be produced next.

    Returns:
        The updated input_ids list, with new tokens appended.
    """
    remaining = target

    while remaining:
        safe_candidates = find_safe_candidates(trie, id_to_token, remaining)

        if not safe_candidates:
            raise ValueError(f"No valid tokens found to generate literal: {target!r}")

        logits = model.get_logits_from_input_ids(input_ids)
        best_token_id = max(safe_candidates, key=lambda token_id: logits[token_id])

        input_ids.append(best_token_id)

        covered_text = id_to_token[best_token_id]
        remaining = remaining[len(covered_text):]

    return input_ids


# ----------BOOLEAN PARAMETER GENERATION----------------

def generate_boolean(
    model: Small_LLM_Model,
    trie: Trie,
    id_to_token: dict[int, str],
    input_ids: list[int],
) -> list[int]:
    """Force the model to generate either 'true' or 'false'.

    Args:
        model: The LLM wrapper.
        trie: The vocabulary trie.
        id_to_token: Reverse mapping from token ID to token string.
        input_ids: Token IDs generated so far (will be extended).

    Returns:
        The updated input_ids list.
    """
    true_candidates = find_safe_candidates(trie, id_to_token, "true")
    false_candidates = find_safe_candidates(trie, id_to_token, "false")

    logits = model.get_logits_from_input_ids(input_ids)
    all_candidates = true_candidates + false_candidates
    best_first_token = max(all_candidates, key=lambda token_id: logits[token_id])

    if best_first_token in true_candidates:
        return generate_literal(model, trie, id_to_token, input_ids, "true")
    else:
        return generate_literal(model, trie, id_to_token, input_ids, "false")


def build_targets(function: dict) -> list:
    """Build the ordered list of JSON pieces needed to represent this function's arguments.

    Each item is either a literal string (fixed punctuation or a quoted key name)
    or a ("VALUE", type) tuple marking a spot where an actual value must be generated.

    Args:
        function: The function definition dict (with "name" and "parameters" keys).

    Returns:
        An ordered list of literal strings and ("VALUE", type) placeholders.
    """
    targets: list = ["{"]
    param_names = list(function["parameters"].keys())

    for i, name in enumerate(param_names):
        targets.append(f'"{name}"')
        targets.append(":")
        param_type = function["parameters"][name]["type"]
        targets.append(("VALUE", param_type))

        if i < len(param_names) - 1:
            targets.append(",")

    targets.append("}")
    return targets


def is_valid_number_continuation(current_number_text: str, token: str) -> bool:
    """Check if appending `token` to `current_number_text` keeps it a valid
    (possibly incomplete) JSON number, allowing an optional leading '-'
    and an optional decimal point.

    Args:
        current_number_text: The number text generated so far for this value.
        token: A candidate token's text to potentially append.

    Returns:
        True if the result would still be a valid (partial) number.
    """
    candidate = current_number_text + token

    if candidate.count("-") > 1:
        return False
    if candidate.count(".") > 1:
        return False

    for i, char in enumerate(candidate):
        if char == "-":
            if i != 0:
                return False
        elif char == ".":
            if i == 0 or candidate[i - 1] == "-":
                return False
        elif not char.isdigit():
            return False

    return True


def get_number_candidate_pool(id_to_token: dict[int, str]) -> list[int]:
    """Find every token ID made only of digits or '.' (a cheap pre-filter;
    position validity is checked separately).

    Args:
        id_to_token: Reverse mapping from token ID to token string.

    Returns:
        A list of token IDs that could plausibly be part of a number.
    """
    number_chars = set("0123456789.-")
    
    pool = []
    for token_id, token in id_to_token.items():
        if token != "" and all(char in number_chars for char in token):
            pool.append(token_id)
    return pool


def generate_number(
    model: Small_LLM_Model,
    trie: Trie,
    id_to_token: dict[int, str],
    input_ids: list[int],
    next_literal: str,
    number_candidate_pool: list[int],
) -> list[int]:
    """Generate a numeric value, optionally with a decimal point, stopping
    once the model prefers the next literal.

    Args:
        model: The LLM wrapper.
        trie: The vocabulary trie.
        id_to_token: Reverse mapping from token ID to token string.
        input_ids: Token IDs generated so far (will be extended).
        next_literal: The literal text that follows this value (e.g. "," or "}").
        number_candidate_pool: Pre-filtered token IDs made only of digits or ".".

    Returns:
        The updated input_ids list.
    """
    stop_candidates = find_safe_candidates(trie, id_to_token, next_literal)
    current_number_text = ""

    while True:
        valid_now = [
            token_id for token_id in number_candidate_pool
            if is_valid_number_continuation(current_number_text, id_to_token[token_id])
        ]
        allowed_token_ids = valid_now + stop_candidates

        logits = model.get_logits_from_input_ids(input_ids)
        best_token_id = max(allowed_token_ids, key=lambda token_id: logits[token_id])

        if best_token_id in stop_candidates:
            break

        input_ids.append(best_token_id)
        current_number_text += id_to_token[best_token_id]

    return input_ids


def get_safe_string_mask(id_to_token: dict[int, str], vocab_size: int) -> np.ndarray:
    """Build a boolean mask (True = safe) for JSON string content tokens
    (excludes raw quotes, backslashes, and control characters).

    Args:
        id_to_token: Reverse mapping from token ID to token string.
        vocab_size: Total number of entries in the model's logits output
            (may be larger than len(id_to_token) due to reserved slots).

    Returns:
        A boolean numpy array of length vocab_size, True where safe.
    """
    mask = np.zeros(vocab_size, dtype=bool)
    unsafe_chars = {'"', "\\"}
    for token_id, token in id_to_token.items():
        if token == "":
            continue
        if any(char in unsafe_chars for char in token):
            continue
        if any(ord(char) < 0x20 for char in token):
            continue
        mask[token_id] = True
    return mask


def generate_string(
    model: Small_LLM_Model,
    trie: Trie,
    id_to_token: dict[int, str],
    input_ids: list[int],
    safe_string_mask: np.ndarray,
    max_string_tokens: int = 50,
) -> list[int]:
    """Generate a string value: opening quote, free content, closing quote."""
    input_ids = generate_literal(model, trie, id_to_token, input_ids, '"')

    quote_candidates = find_safe_candidates(trie, id_to_token, '"')
    allowed_mask = safe_string_mask.copy()
    allowed_mask[quote_candidates] = True

    for _ in range(max_string_tokens):
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        masked_logits = np.where(allowed_mask, logits, -np.inf)
        best_token_id = int(np.argmax(masked_logits))

        if best_token_id in quote_candidates:
            break

        input_ids.append(best_token_id)

    input_ids = generate_literal(model, trie, id_to_token, input_ids, '"')
    return input_ids


def generate_from_targets(
    model: Small_LLM_Model,
    trie: Trie,
    id_to_token: dict[int, str],
    input_ids: list[int],
    targets: list,
    number_candidate_pool: list[int],
    safe_string_mask: np.ndarray,
) -> list[int]:
    """Walk the targets list, generating literals and values in order.

    Args:
        model: The LLM wrapper.
        trie: The vocabulary trie.
        id_to_token: Reverse mapping from token ID to token string.
        input_ids: Token IDs generated so far (will be extended).
        targets: The ordered list from build_targets.
        number_candidate_pool: Pre-filtered token IDs made only of digits or ".".
        safe_string_mask: Boolean mask of token IDs safe for string content.

    Returns:
        The updated input_ids list.
    """
    for i, target in enumerate(targets):
        if isinstance(target, tuple):
            value_type = target[1]
            if value_type == "number":
                next_literal = targets[i + 1]
                input_ids = generate_number(
                    model, trie, id_to_token, input_ids, next_literal, number_candidate_pool
                )
            elif value_type == "string":
                input_ids = generate_string(
                    model, trie, id_to_token, input_ids, safe_string_mask
                )
            elif value_type == "boolean":
                input_ids = generate_boolean(model, trie, id_to_token, input_ids)
            continue

        input_ids = generate_literal(model, trie, id_to_token, input_ids, target)

    return input_ids
