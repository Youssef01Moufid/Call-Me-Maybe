from llm_sdk.llm_sdk import Small_LLM_Model
import json
from constraint_decoder import ft_tokinize_function

def main() -> None:

    llm_model = Small_LLM_Model()
    with open("/home/ymoufid/call_me_maybe/data/input/functions_definition.json") as f:
        functions = json.load(f)

    tokens_func = ft_tokinize_function(llm_model, functions)
    print(tokens_func)

    logits_of_tokens = llm_model.get_logits_from_input_ids(tokens_func[0])
    print(len(logits_of_tokens))



if __name__ == "__main__":
    main()