from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
import json
import time
import argparse
import os
from .functions_validation import create_dynamic_validators, load_functions
from .generation import (get_token_masks, force_sequence,
                         generate_pydantic_object)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constraint JSON generator for LLM")
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to JSON file containing the prompts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Destination path for the generated results"
    )
    args = parser.parse_args()

    model = Small_LLM_Model()

    with open(args.input, 'r', encoding='utf-8') as f:
        prompts_list = json.load(f)
    with open(model.get_path_to_vocab_file(), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open(
         "data/input/functions_definition.json", "r", encoding="utf-8") as f:
        system_context = f.read()

    valid_number_token_ids, valid_string_token_ids, closing_quote_token_ids, \
        valid_bool_token_ids = get_token_masks(vocab)
    functions_list = load_functions("data/input/functions_definition.json")
    functions_validators = create_dynamic_validators(functions_list)
    valid_function_sequences = [
        model.encode(func["name"])[0].tolist() for func in functions_list
    ]
    id_open_brace = vocab.get("{")
    results_list = []

    print("\nStarting generation...")
    global_start_time = time.perf_counter()

    for prompt in prompts_list:
        prompt_text = prompt["prompt"]
        print(f"\nPrompt: '{prompt_text}'")

        full_prompt = (
            f"Available functions:\n{system_context}\n\n"
            f"User request: {prompt_text}\n"
        )
        input_ids_list = model.encode(full_prompt)[0].tolist()
        initial_prompt_length = len(input_ids_list)

        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[id_open_brace] = logits[id_open_brace]
        best_token_id = int(np.argmax(logits_contraints))
        input_ids_list.append(best_token_id)
        print(model.decode([best_token_id]), end="", flush=True)

        force_sequence(model, input_ids_list, '"name": "')

        active_candidates = valid_function_sequences.copy()
        name_token_index = 0
        while True:
            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_contraints = np.full_like(logits, -np.inf)
            allowed_tokens = {
                seq[name_token_index] for seq in active_candidates
                if name_token_index < len(seq)
            }
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

        force_sequence(model, input_ids_list, '", "parameters": ')

        current_function = model.decode(active_candidates[0])
        current_validator = functions_validators[current_function]

        generate_pydantic_object(
            model, input_ids_list, vocab, current_validator,
            valid_number_token_ids, valid_string_token_ids,
            closing_quote_token_ids, valid_bool_token_ids
        )

        force_sequence(model, input_ids_list, '}')
        print()

        generated_json_string = model.decode(input_ids_list[
            initial_prompt_length:])
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

    global_end_time = time.perf_counter()
    total_time = global_end_time - global_start_time
    mins, secs = divmod(total_time, 60)
    print(f"Global generation done in {int(mins)}m {secs:.1f}s")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(results_list, out_file, indent=4)


if __name__ == "__main__":
    main()
