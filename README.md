*This project has been created as part of the 42 curriculum by <ymoufid>.*

# call me maybe

## Description

This project implements a function-calling system that converts natural language prompts into structured, schema-compliant JSON function calls, using a small local language model (Qwen/Qwen3-0.6B). Given a prompt like *"What is the sum of 2 and 3?"*, the program does not answer the question directly — instead it outputs which function to call and with what arguments, e.g. `fn_add_numbers` with `{"a": 2, "b": 3}`.

The core challenge this project solves is reliability: small language models are notoriously unreliable at producing valid structured output when simply prompted to do so. This project solves that with **constrained decoding**, implemented from scratch at the token level — the model's output is masked at every generation step so that only tokens consistent with valid JSON and the target function's schema can ever be produced. This guarantees 100% valid, schema-compliant JSON output regardless of the model's own reliability.

## Instructions

### Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Install

```bash
make install
```

This runs `uv sync`, installing all dependencies declared in `pyproject.toml`/`uv.lock`, including the `llm_sdk` package (installed as a local editable dependency).

### Run

```bash
make run
```

or directly:

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

By default, the program reads from `data/input/functions_definition.json` and `data/input/function_calling_tests.json`, and writes results to `data/output/function_calling_results.json`.

### Other Makefile targets

- `make debug` — run the program under Python's built-in debugger (`pdb`)
- `make clean` — remove `__pycache__` and cache directories
- `make lint` — run `flake8` and `mypy` with the required flags
- `make lint-strict` — run `flake8` and `mypy --strict` (optional, stricter checking)

## Resources

- [Qwen3 model card and tokenizer documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Trie (prefix tree) data structure](https://en.wikipedia.org/wiki/Trie)
- General background on constrained/guided decoding for structured LLM output

### AI usage disclosure

Claude (Anthropic) was used throughout this project as a learning and pair-programming aid, specifically for:
- Explaining core concepts (tokenization, logits, trie data structures, finite-state-machine-style constrained generation) before implementation, with worked examples
- Reviewing and debugging code written by the author (e.g. catching a `mask_logits`/`masked_logits` naming bug that caused an infinite loop, a numpy array shape mismatch between the vocab file's token count and the model's actual logits output size, and a stale-path `uv` packaging issue after the project directory was moved)
- Suggesting the fix for unreliable prompt understanding: reformatting raw prompts into instruction-style prompts before encoding, which measurably improved both function-selection accuracy (documented below) and argument-extraction accuracy
- Proposing the safety-net design for open-ended generation (numeric and string values), including the max-length cutoff to guarantee the program never hangs or crashes on a single difficult prompt

All generated code was reviewed, tested, and understood by the author before inclusion; no code was used without being explained first.

## Algorithm explanation

The program does not ask the model to freely generate a whole answer. Instead, generation is broken into small, individually-constrained steps:

1. **Function selection**: every candidate function name is pre-tokenized. At each generation step, the set of "still possible" candidate names is narrowed based on which token the model actually chose, until exactly one candidate remains and has been fully generated. This guarantees the selected name is always one of the valid function names — never an invented one.

2. **Target planning**: once a function is selected, its parameter schema (from `functions_definition.json`) is used to build an ordered list of "targets" — a mix of known literal text (`{`, `"paramName"`, `:`, `,`, `}`) and placeholders marking where a real value (of a known type) must be generated.

3. **Literal generation**: for each literal target, a vocabulary trie is used to find every token whose text is a valid, non-overshooting prefix of the remaining text still needed. The model's logits are restricted to only these candidates, guaranteeing the exact literal text is spelled out correctly regardless of tokenization boundaries (a single literal may take one or several tokens to produce).

4. **Value generation**:
   - **Numbers**: only tokens made purely of ASCII digits are allowed at each step, alongside tokens that would start the next known literal (e.g. `,` or `}`). Whichever the model prefers determines whether the number continues or is complete.
   - **Strings**: the opening and closing `"` characters are generated as literals; the content in between is generated freely but restricted to characters safe for a JSON string (excluding raw quotes, backslashes, and control characters), with a maximum-length safety cutoff to guarantee the program cannot hang indefinitely on a difficult value.

5. **Function name and parameters are combined into the final `{"prompt", "name", "parameters"}` result and written as a JSON array.**

This approach guarantees valid, schema-compliant JSON is produced on every single token, because invalid tokens are never made available for selection in the first place — the model is never in a position to "choose" something structurally wrong.

## Design decisions

- **Vocabulary trie over a linear scan**: a trie was built from the model's full ~151k-token vocabulary to make "which tokens match this prefix" queries fast, rather than scanning the whole vocabulary on every generation step.
- **Instruction-style prompts instead of raw prompts**: initial testing showed that feeding the model the raw user prompt directly resulted in very poor function selection and value extraction (the model would simply continue the text rather than answer). Wrapping the prompt with a short instruction describing the task measurably fixed this.
- **`max()` over pre-filtered candidates instead of masking the full logits array with Python loops**: an early version masked the entire ~151k-length logits array using a Python `for` loop on every generation step, which was a major performance bottleneck. This was replaced with either (a) taking `max()` directly over a small pre-filtered candidate list, or (b) a vectorized numpy boolean mask for larger candidate sets (string content generation), both of which are substantially faster.
- **A safety cutoff for open-ended generation (strings)**: rather than looping forever (or crashing) if the model never "chooses" to close a string, string generation force-closes after a maximum token count, guaranteeing the program always terminates and always produces valid JSON, even in a worst case.
- **Per-prompt error isolation**: each prompt is processed inside its own `try/except` block. A single prompt failing to generate correctly is logged and skipped, rather than crashing the entire batch — directly satisfying the requirement that the program must never crash unexpectedly.

## Performance analysis

- **JSON validity**: 100% of generated outputs are valid, parseable JSON matching the target function's schema, by construction (constrained decoding never allows an invalid token to be selected).
- **Function selection accuracy**: after switching to instruction-formatted prompts, the model correctly selected the right function for all 12 prompts in the test set (100% on the tested sample), up from a majority-correct baseline with raw prompts.
- **Argument extraction accuracy**: number and string extraction were correct in the large majority of tested cases. One known weakness: the model occasionally extracts a *computed* value rather than the *input* value for prompts implying a calculation (e.g. extracting the answer to a square root rather than the input to be square-rooted), and can occasionally struggle to extract the first of two numbers correctly in an arithmetic prompt. This is a genuine small-model reasoning limitation rather than a decoding bug, and falls within the project's 90%+ (not 100%) accuracy tolerance for value extraction.
- **Speed**: after replacing full-vocabulary Python-loop masking with targeted candidate lists and vectorized numpy masking, all test prompts process well within the 5-minute budget.

## Challenges faced

- **Infinite loop from a naming bug**: `mask_logits` (the function) was accidentally passed to `np.argmax` instead of `masked_logits` (the variable), causing the same token to be picked forever. Fixed by carefully re-reading variable names, since Python did not raise an error for this mistake.
- **Numpy shape mismatch**: the vocabulary file's token count (`len(vocab)`) was smaller than the model's actual logits output length, causing a broadcasting error when masking. Fixed by sizing the mask from a real sample logits call instead of the vocab file's length.
- **Unreliable value extraction from raw prompts**: the model would frequently ignore the actual numbers/content in the prompt when given no task framing. Fixed with instruction-style prompt formatting.
- **Runaway string generation**: for a difficult prompt (asking the model to generate a regex pattern), string generation occasionally got stuck in a repetitive loop and never chose to close the string. Fixed with a maximum-token safety cutoff that force-closes the string rather than crashing.
- **`uv` packaging and environment issues**: moving the project directory broke an editable install of `llm_sdk`, and `uv`'s default packaging assumptions (`src/<project_name>/__init__.py`) conflicted with this project's flat `src/` layout. Fixed by setting `package = false` under `[tool.uv]` in `pyproject.toml` and re-adding `llm_sdk` as an editable dependency from the new path.

## Testing strategy

- Each core function (trie insertion/search, literal generation, number generation, string generation, function selection) was manually tested in isolation with small, hand-inspectable examples before being wired into the full pipeline.
- The full pipeline was run end-to-end against the provided example `functions_definition.json` and `function_calling_tests.json`, with output manually inspected for JSON validity and semantic correctness against each prompt.
- Deliberate edge cases were tested: a missing input file, a malformed JSON input file, and a prompt requiring open-ended (regex) string generation likely to stress the safety cutoff.

## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example input (`function_calling_tests.json`):
```json
[{ "prompt": "What is the sum of 2 and 3?" }]
```

Example output (`function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  }
]
```