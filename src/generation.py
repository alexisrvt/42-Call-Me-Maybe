import numpy as np
from typing import Any, get_origin, get_args
from pydantic import BaseModel


def get_token_masks(vocab: dict[str, int]) -> tuple[list[int], list[int],
                                                    list[int], list[int]]:
    """
    Pre-calculates valid token IDs for specific JSON data types.

    Args:
        vocab (dict[str, int]): The BPE vocabulary dictionary mapping strings
        to token IDs.

    Returns:
        tuple[list[int], list[int], list[int], list[int]]: A tuple containing
        four lists of valid token IDs:
            - Numbers (digits, minus sign, period)
            - Strings (excluding invalid whitespace characters)
            - Closing quotes
            - Booleans ('true' and 'false')
    """
    valid_number_chars = set("0123456789.-")
    valid_number_token_ids = {
        tid for tstr, tid in vocab.items()
        if len(tstr) > 0 and all(c in valid_number_chars for c in tstr)
    }

    invalid_string_chars = set("\n\r\t")
    valid_string_token_ids = set()
    closing_quote_token_ids = set()
    for tstr, tid in vocab.items():
        if any(c in invalid_string_chars for c in tstr) or "**" in tstr:
            continue

        if '"' in tstr:
            if tstr.startswith('"'):
                closing_quote_token_ids.add(tid)
            elif (
                '"' not in tstr.replace('\\"', '')
                or tstr in ('Ġ"', ' "')
            ):
                valid_string_token_ids.add(tid)
        else:
            valid_string_token_ids.add(tid)

    valid_bool_token_ids: set[int] = {
        tid for tstr, tid in vocab.items()
        if tstr.strip() in ("true", "false")
    }

    return (
        list(valid_number_token_ids), list(valid_string_token_ids),
        list(closing_quote_token_ids), list(valid_bool_token_ids)
    )


def force_sequence(model: Any, input_ids_list: list[int], text: str) -> None:
    """
    Forces the model to output an exact string sequence without computing
    logits.

    Args:
        model (Any): The LLM instance containing the tokenizer.
        input_ids_list (list[int]): The current list of generated token IDs
        (modified in place).
        text (str): The exact string sequence to inject.

    Returns:
        None
    """
    target_ids = model.encode(text)[0].tolist()
    for expected_token in target_ids:
        input_ids_list.append(expected_token)
    print(text, end="", flush=True)


def generate_string(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        valid_string_token_ids: list[int],
        closing_quote_token_ids: list[int]) -> None:
    """
    Generates a constrained JSON string value.

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        vocab (dict[str, int]): The tokenizer vocabulary.
        valid_string_token_ids (list[int]): Pre-calculated valid string tokens.
        closing_quote_token_ids (list[int]): Pre-calculated tokens containing
        closing quotes.

    Returns:
        None
    """
    id_quote = int(vocab['"'])
    input_ids_list.append(id_quote)
    print('"', end="", flush=True)

    backslash_ids = [
        tid for tstr, tid in vocab.items()
        if "\\" in tstr or '"' in tstr
    ]
    asterisk_ids = [
        tid for tstr, tid in vocab.items()
        if "*" in tstr
    ]

    token_count = 0
    max_tokens = 40
    generated_str = ""

    while True:
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_constraints = np.full_like(logits, -np.inf)
        logits_constraints[valid_string_token_ids] = logits[
            valid_string_token_ids]
        if generated_str.count('\\"') >= 2:
            logits_constraints[backslash_ids] = -np.inf
        if generated_str.endswith("*"):
            logits_constraints[asterisk_ids] = -np.inf
        if token_count > 0:
            logits_constraints[closing_quote_token_ids] = logits[
                closing_quote_token_ids]
        best_token_id = int(np.argmax(logits_constraints))
        token_count += 1

        if best_token_id in closing_quote_token_ids or \
                token_count >= max_tokens:
            odd_backslashes = (
                len(generated_str) - len(generated_str.rstrip("\\"))
            ) % 2 == 1
            if token_count < max_tokens and odd_backslashes:
                input_ids_list.append(id_quote)
                generated_str += '"'
                print('"', end="", flush=True)
                continue
            if odd_backslashes:
                force_sequence(model, input_ids_list, "\\")
            input_ids_list.append(id_quote)
            print('"', end="", flush=True)
            break

        token_str = model.decode([best_token_id])
        if token_str == ' "':
            escaped_quote = '\\"' if token_count == 1 else ' \\"'
            force_sequence(model, input_ids_list, escaped_quote)
            generated_str += escaped_quote
            continue

        if token_count == 1 and token_str.startswith(" "):
            clean_str = token_str.lstrip(" ")
            if clean_str:
                force_sequence(model, input_ids_list, clean_str)
                generated_str += clean_str
                continue

        input_ids_list.append(best_token_id)
        generated_str += token_str
        print(token_str, end="", flush=True)


def generate_number(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        valid_number_token_ids: list[int], is_float: bool = False) -> None:
    """
    Generates a constrained JSON numeric value (integer or float).

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        vocab (dict[str, int]): The tokenizer vocabulary.
        valid_number_token_ids (list[int]): Pre-calculated valid numeric
        tokens.
        is_float (bool, optional): Whether to enforce a decimal point (.0)
        for float fields. Defaults to False.

    Returns:
        None
    """
    id_comma = int(vocab[","])
    id_close_brace = int(vocab["}"])
    id_close_bracket = int(vocab["]"])

    token_count = 0
    max_tokens = 15
    generated_str = ""

    while True:
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_constraints = np.full_like(logits, -np.inf)
        logits_constraints[valid_number_token_ids] = logits[
            valid_number_token_ids]
        if token_count > 0 and not generated_str.endswith((".", "-")):
            logits_constraints[
                [id_comma, id_close_brace, id_close_bracket]] = logits[
                [id_comma, id_close_brace, id_close_bracket]]
        best_token_id = int(np.argmax(logits_constraints))
        token_count += 1

        if best_token_id in (id_comma, id_close_brace, id_close_bracket) or \
                token_count >= max_tokens:
            if is_float and "." not in generated_str:
                force_sequence(model, input_ids_list, ".0")
            break

        input_ids_list.append(best_token_id)
        token_str = model.decode([best_token_id])
        generated_str += token_str
        print(token_str, end="", flush=True)


def generate_boolean(
        model: Any, input_ids_list: list[int],
        valid_bool_token_ids: list[int]) -> None:
    """
    Generates a constrained JSON boolean value (true/false).

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        valid_bool_token_ids (list[int]): Pre-calculated boolean tokens.

    Returns:
        None
    """
    logits = np.array(model.get_logits_from_input_ids(input_ids_list))
    logits_constraints = np.full_like(logits, -np.inf)
    logits_constraints[valid_bool_token_ids] = logits[valid_bool_token_ids]
    best_token_id = int(np.argmax(logits_constraints))
    input_ids_list.append(best_token_id)
    print(model.decode([best_token_id]), end="", flush=True)


def generate_null(model: Any, input_ids_list: list[int]) -> None:
    """
    Forces the generation of a JSON null value.

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.

    Returns:
        None
    """
    force_sequence(model, input_ids_list, "null")


def generate_list(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        inner_type: type, valid_string_token_ids: list[int],
        valid_number_token_ids: list[int], closing_quote_token_ids: list[int],
        valid_bool_token_ids: list[int]) -> None:
    """
    Generates a constrained JSON array, recursively processing internal items.

    Applies a dynamic probability malus to the closing bracket token ']' based
    on the structural complexity of the internal type to prevent early
    stopping.

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        vocab (dict[str, int]): The tokenizer vocabulary.
        inner_type (type): The expected Python type of the array items.
        valid_string_token_ids (list[int]): Valid string tokens.
        valid_number_token_ids (list[int]): Valid numeric tokens.
        closing_quote_token_ids (list[int]): Valid closing quote tokens.
        valid_bool_token_ids (list[int]): Valid boolean tokens.

    Returns:
        None
    """
    id_open, id_close, id_comma_simple = int(vocab["["]), int(vocab["]"]), \
        int(vocab[","])
    id_comma_space = model.encode(", ")[0].tolist()[0]
    input_ids_list.append(id_open)
    print("[", end="", flush=True)

    max_items = 20
    item_count = 0

    is_complex = isinstance(inner_type, type) and issubclass(inner_type,
                                                             BaseModel)
    malus = 10.0 if is_complex else 1.0

    while True:
        generate_value(model, input_ids_list, vocab, inner_type,
                       valid_number_token_ids, valid_string_token_ids,
                       closing_quote_token_ids, valid_bool_token_ids)
        item_count += 1

        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_constraints = np.full_like(logits, -np.inf)
        logits_constraints[id_comma_simple] = logits[id_comma_simple]
        logits_constraints[id_comma_space] = logits[id_comma_space]
        logits_constraints[id_close] = logits[id_close] - malus
        best_token_id = int(np.argmax(logits_constraints))
        if best_token_id == id_close or item_count >= max_items:
            force_sequence(model, input_ids_list, "]")
            break
        force_sequence(model, input_ids_list, ", ")


def generate_value(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        expected_type: type, valid_number_token_ids: list[int],
        valid_string_token_ids: list[int], closing_quote_token_ids: list[int],
        valid_bool_token_ids: list[int]) -> None:
    """
    Recursive router that delegates generation to the appropriate type-
    specific function.

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        vocab (dict[str, int]): The tokenizer vocabulary.
        expected_type (type): The target Python type/Pydantic model to
        generate.
        valid_number_token_ids (list[int]): Valid numeric tokens.
        valid_string_token_ids (list[int]): Valid string tokens.
        closing_quote_token_ids (list[int]): Valid closing quote tokens.
        valid_bool_token_ids (list[int]): Valid boolean tokens.

    Returns:
        None
    """
    origin_type = get_origin(expected_type) or expected_type
    if origin_type in (float, int):
        is_float = (origin_type is float)
        generate_number(
            model, input_ids_list, vocab, valid_number_token_ids, is_float)
    elif origin_type == str:
        generate_string(
            model, input_ids_list, vocab, valid_string_token_ids,
            closing_quote_token_ids)
    elif origin_type == bool:
        generate_boolean(model, input_ids_list, valid_bool_token_ids)
    elif origin_type is type(None):
        generate_null(model, input_ids_list)
    elif origin_type is list:
        inner_type = get_args(expected_type)[0]
        generate_list(
            model, input_ids_list, vocab, inner_type, valid_string_token_ids,
            valid_number_token_ids, closing_quote_token_ids,
            valid_bool_token_ids)
    elif isinstance(origin_type, type) and issubclass(origin_type, BaseModel):
        generate_pydantic_object(
            model, input_ids_list, vocab, origin_type, valid_number_token_ids,
            valid_string_token_ids, closing_quote_token_ids,
            valid_bool_token_ids)


def generate_pydantic_object(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        pydantic_class: type[BaseModel], valid_number_token_ids: list[int],
        valid_string_token_ids: list[int], closing_quote_token_ids: list[int],
        valid_bool_token_ids: list[int]) -> None:
    """
    Generates a constrained JSON object corresponding to a Pydantic model.

    Iterates through the model's fields, injecting keys directly via
    `force_sequence` and recursively calling `generate_value` for field
    values.

    Args:
        model (Any): The LLM instance.
        input_ids_list (list[int]): The current list of generated token IDs.
        vocab (dict[str, int]): The tokenizer vocabulary.
        pydantic_class (type[BaseModel]): The Pydantic model class dictating
        the object schema.
        valid_number_token_ids (list[int]): Valid numeric tokens.
        valid_string_token_ids (list[int]): Valid string tokens.
        closing_quote_token_ids (list[int]): Valid closing quote tokens.
        valid_bool_token_ids (list[int]): Valid boolean tokens.

    Returns:
        None
    """
    force_sequence(model, input_ids_list, "{")
    is_first = True

    for field_name, field_info in pydantic_class.model_fields.items():
        key_str = f'"{field_name}": ' if is_first else f', "{field_name}": '
        is_first = False
        force_sequence(model, input_ids_list, key_str)
        expected_type = field_info.annotation or str
        generate_value(
            model, input_ids_list, vocab, expected_type,
            valid_number_token_ids, valid_string_token_ids,
            closing_quote_token_ids, valid_bool_token_ids)
    force_sequence(model, input_ids_list, "}")
