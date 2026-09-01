from llm_sdk import Small_LLM_Model
import numpy as np
import json
import os
from functions_validation import create_dynamic_validators, load_functions


def main() -> None:
    model = Small_LLM_Model()

    prompts_path = "data/input/function_calling_tests.json"
    with open(prompts_path, 'r') as f:
        prompts_list = json.load(f)

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    valid_number_chars = set("01234567890.-")
    valid_number_token_ids = set()
    for token_str, token_id in vocab.items():
        if len(token_str) > 0 and all(char in valid_number_chars for char in token_str):
            valid_number_token_ids.add(token_id)

    invalid_string_chars = set("\n\r\t")
    valid_string_token_ids = set()
    for token_str, token_id in vocab.items():
        if not any(char in invalid_string_chars for char in token_str):
            valid_string_token_ids.add(token_id)

    functions_list = load_functions("data/input/functions_definition.json")
    functions_validators = create_dynamic_validators(functions_list)
    id_open_brace = vocab.get("{")
    id_close_brace = vocab.get("}")
    valid_function_sequences = [model.encode(func.name)[0].tolist() for func in functions_list]

    results_list = []

    for prompt in prompts_list:
        best_token_id = -1
        prompt_text = prompt["prompt"]
        print(f"Prompt: '{prompt_text}'")
        input_ids = model.encode(prompt_text)

        input_ids_list = input_ids[0].tolist()
        initial_prompt_length = len(input_ids_list)
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[id_open_brace] = logits[id_open_brace]
        best_token_id = int(np.argmax(logits_contraints))
        input_ids_list.append(best_token_id)
        print(model.decode([best_token_id]), end="", flush=True)
        target_ids = model.encode('"name": "')[0].tolist()
        for expected_tokens in target_ids:
            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_contraints = np.full_like(logits, -np.inf)
            logits_contraints[expected_tokens] = logits[expected_tokens]
            best_token_id = int(np.argmax(logits_contraints))
            input_ids_list.append(best_token_id)
            print(model.decode([best_token_id]), end="", flush=True)
        active_candidates = valid_function_sequences.copy()
        name_token_index = 0
        while True:
            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_contraints = np.full_like(logits, -np.inf)
            allowed_tokens = set()
            for seq in active_candidates:
                if name_token_index < len(seq):
                    allowed_tokens.add(seq[name_token_index])
            if not allowed_tokens:
                break
            for t in allowed_tokens:
                logits_contraints[t] = logits[t]
            best_token_id = int(np.argmax(logits_contraints))
            input_ids_list.append(best_token_id)
            print(model.decode([best_token_id]), end="", flush=True)
            active_candidates = [
                seq for seq in active_candidates
                if name_token_index < len(seq)
                and seq[name_token_index] == best_token_id
            ]
            name_token_index += 1

        target_ids = model.encode('", "parameters": {')[0].tolist()
        for expected_tokens in target_ids:
            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_contraints = np.full_like(logits, -np.inf)
            logits_contraints[expected_tokens] = logits[expected_tokens]
            best_token_id = int(np.argmax(logits_contraints))
            input_ids_list.append(best_token_id)
            print(model.decode([best_token_id]), end="", flush=True)
        current_function = model.decode(active_candidates[0])
        current_validator = functions_validators[current_function]
        is_first_param = True
        for field_name, field_info in current_validator.model_fields.items():
            if is_first_param:
                key_str = f'"{field_name}": '
                is_first_param = False
            else:
                key_str = f', "{field_name}": '

            target_ids = model.encode(key_str)[0].tolist()
            for expected_tokens in target_ids:
                logits = np.array(model.get_logits_from_input_ids(input_ids_list))
                logits_contraints = np.full_like(logits, -np.inf)
                logits_contraints[expected_tokens] = logits[expected_tokens]
                best_token_id = int(np.argmax(logits_contraints))
                input_ids_list.append(best_token_id)
                print(model.decode([best_token_id]), end="", flush=True)
            expected_type = field_info.annotation
            if expected_type == float:
                id_comma = vocab.get(",")
                while True:
                    logits = np.array(model.get_logits_from_input_ids(input_ids_list))
                    logits_contraints = np.full_like(logits, -np.inf)
                    for t in valid_number_token_ids:
                        logits_contraints[t] = logits[t]
                    if id_comma is not None:
                        logits_contraints[id_comma] = logits[id_comma]
                    if id_close_brace is not None:
                        logits_contraints[id_close_brace] = logits[id_close_brace]
                    best_token_id = int(np.argmax(logits_contraints))
                    if best_token_id in (id_comma, id_close_brace):
                        break
                    input_ids_list.append(best_token_id)
                    print(model.decode([best_token_id]), end="", flush=True)
            elif expected_type == str:
                id_quote = vocab.get('"')
                input_ids_list.append(id_quote)
                print('"', end="", flush=True)
                while True:
                    logits = np.array(model.get_logits_from_input_ids(input_ids_list))
                    logits_contraints = np.full_like(logits, -np.inf)
                    for t in valid_string_token_ids:
                        logits_contraints[t] = logits[t]
                    best_token_id = int(np.argmax(logits_contraints))
                    if best_token_id == id_quote:
                        input_ids_list.append(best_token_id)
                        print('"', end="", flush=True)
                        break
                    input_ids_list.append(best_token_id)
                    print(model.decode([best_token_id]), end="", flush=True)
        target_ids = model.encode('}}')[0].tolist()
        for expected_tokens in target_ids:
            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_contraints = np.full_like(logits, -np.inf)
            logits_contraints[expected_tokens] = logits[expected_tokens]
            best_token_id = int(np.argmax(logits_contraints))
            input_ids_list.append(best_token_id)
            print(model.decode([best_token_id]), end="", flush=True)
        print()

        generated_tokens = input_ids_list[initial_prompt_length:]
        generated_json_string = model.decode(generated_tokens)
        try:
            result_dict = json.loads(generated_json_string)
            final_obj = {
                "prompt": prompt_text,
                "name": result_dict["name"],
                "parameters": result_dict["parameters"]
            }
            results_list.append(final_obj)
        except json.JSONDecodeError:
            print(f"\n Parsing error for the prompt: {prompt_text}")
    os.makedirs("data/output", exist_ok=True)
    output_path = "data/output/function_calling_results.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results_list, out_file, indent=4)


if __name__ == "__main__":
    main()
