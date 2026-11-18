from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
import json
import time
import argparse
import os
from .functions_validation import create_dynamic_validators, load_functions
from .generation import (get_token_masks, force_sequence,
                         generate_pydantic_object)


def existing_json_file(path: str) -> str:
    """
    Argparse type validator ensuring an input file exists and is a JSON file.

    Args:
        path (str): The file path provided via CLI.

    Raises:
        argparse.ArgumentTypeError: If the file does not exist or lacks a
        .json extension.

    Returns:
        str: The validated file path.
    """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"The file '{path}' cannot be found.")
    if not path.endswith('.json'):
        raise argparse.ArgumentTypeError(
            f"The input file '{path}' must have the .json extension."
        )
    return path


def valid_output_path(path: str) -> str:
    """
    Argparse type validator ensuring the output path targets a JSON file.

    Args:
        path (str): The file path provided via CLI.

    Raises:
        argparse.ArgumentTypeError: If the file lacks a .json extension.

    Returns:
        str: The validated file path.
    """
    if not path.endswith('.json'):
        raise argparse.ArgumentTypeError(
            f"The output file '{path}' must have the .json extension"
        )
    return path


def main() -> None:
    """
    Main entry point for the Call Me Maybe constrained JSON generator.

    Parses CLI arguments to load system contexts, vocabulary, and user prompts.
    Applies strict logit constraints to the LLM generation process to extract
    100% compliant JSON structures aligned with dynamically loaded Pydantic
    schemas.
    Outputs the extracted function calls to the specified JSON destination.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Constraint JSON generator for LLM")
    parser.add_argument(
        "--functions_definition",
        type=existing_json_file,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing the functions definition"
    )
    parser.add_argument(
        "--input",
        type=existing_json_file,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing the prompts"
    )
    parser.add_argument(
        "--output",
        type=valid_output_path,
        default="data/output/function_calling_results.json",
        help="Destination path for the generated results"
    )
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            prompts_list = json.load(f)
        with open(
             args.functions_definition, "r", encoding="utf-8") as f:
            system_context = f.read()
        functions_list = load_functions(args.functions_definition)
    except (json.JSONDecodeError, OSError) as e:
        parser.error(f"Error reading JSON file: {e}")

    if (
        not isinstance(prompts_list, list)
        or not prompts_list
        or not all(
            isinstance(p, dict) and isinstance(p.get("prompt"), str)
            for p in prompts_list
        )
    ):
        parser.error(
            "Input JSON file must contain a non-empty list of objects "
            "with a valid 'prompt' string."
        )

    if (
        not isinstance(functions_list, list)
        or not functions_list
        or not all(
            isinstance(fn, dict)
            and isinstance(fn.get("name"), str)
            and bool(fn.get("name"))
            and isinstance(fn.get("parameters", {}), dict)
            for fn in functions_list
        )
    ):
        parser.error(
            "Functions JSON file must contain a non-empty list of valid "
            "function definitions with a 'name' string."
        )

    try:
        functions_validators = create_dynamic_validators(functions_list)
    except Exception as e:
        parser.error(f"Invalid function schema definition: {e}")

    model = Small_LLM_Model()
    with open(model.get_path_to_vocab_file(), 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    valid_number_token_ids, valid_string_token_ids, closing_quote_token_ids, \
        valid_bool_token_ids = get_token_masks(vocab)
    valid_function_sequences = [
        model.encode(func["name"])[0].tolist() for func in functions_list
    ]
    id_open_brace = vocab.get("{")
    results_list = []

    print("\nStarting generation...")
    global_start_time = time.perf_counter()

    try:
        for prompt in prompts_list:
            prompt_text = prompt["prompt"]
            print(f"\nPrompt: '{prompt_text}'")

            sanitized_prompt = (
                prompt_text.replace('"', '\\"')
                if ": " in prompt_text else prompt_text
            )
            full_prompt = (
                "You are a strict and precise data extraction AI. Your task is"
                " to extract ALL requested data into the JSON format.\n"
                "CRITICAL: When extracting arrays or lists, you MUST include "
                "EVERY single item mentioned in the user request. "
                "Do not stop early. Do not invent, hallucinate, or add extra "
                "items. Be exhaustive.\n\n"
                f"Available functions:\n{system_context}\n\n"
                f"User request: {sanitized_prompt}\n"
            )
            input_ids_list = model.encode(full_prompt)[0].tolist()
            initial_prompt_length = len(input_ids_list)

            logits = np.array(model.get_logits_from_input_ids(input_ids_list))
            logits_constraints = np.full_like(logits, -np.inf)
            logits_constraints[id_open_brace] = logits[id_open_brace]
            best_token_id = int(np.argmax(logits_constraints))
            input_ids_list.append(best_token_id)
            print(model.decode([best_token_id]), end="", flush=True)

            force_sequence(model, input_ids_list, '"name": "')

            active_candidates = valid_function_sequences.copy()
            name_token_index = 0
            while True:
                logits = np.array(
                    model.get_logits_from_input_ids(input_ids_list))
                logits_constraints = np.full_like(logits, -np.inf)
                allowed_tokens = {
                    seq[name_token_index] for seq in active_candidates
                    if name_token_index < len(seq)
                }
                if not allowed_tokens:
                    break
                for t in allowed_tokens:
                    logits_constraints[t] = logits[t]

                best_token_id = int(np.argmax(logits_constraints))
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
    except KeyboardInterrupt:
        print("\n\nGeneration manually interrupted. Saving partial results...")

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
