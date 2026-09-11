import numpy as np
from typing import Any, get_origin, get_args
from pydantic import BaseModel


def get_token_masks(vocab: dict[str, int]) -> tuple[list[int], list[int],
                                                    list[int], list[int]]:
    valid_number_chars = set("0123456789.-")
    valid_number_token_ids = {
        tid for tstr, tid in vocab.items()
        if len(tstr) > 0 and all(c in valid_number_chars for c in tstr)
    }

    invalid_string_chars = set("\n\r\t")
    valid_string_token_ids = set()
    closing_quote_token_ids = set()
    for tstr, tid in vocab.items():
        if any(c in invalid_string_chars for c in tstr):
            continue

        if '"' in tstr:
            if tstr.startswith('"'):
                closing_quote_token_ids.add(tid)
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
    target_ids = model.encode(text)[0].tolist()
    for expected_token in target_ids:
        input_ids_list.append(expected_token)
    print(text, end="", flush=True)


def generate_string(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        valid_string_token_ids: list[int],
        closing_quote_token_ids: list[int]) -> None:
    id_quote = int(vocab['"'])
    input_ids_list.append(id_quote)
    print('"', end="", flush=True)

    token_count = 0
    max_tokens = 40

    while True:
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[valid_string_token_ids] = logits[
            valid_string_token_ids]
        logits_contraints[closing_quote_token_ids] = logits[
            closing_quote_token_ids]
        best_token_id = int(np.argmax(logits_contraints))
        token_count += 1

        if best_token_id in closing_quote_token_ids or \
                token_count >= max_tokens:
            input_ids_list.append(id_quote)
            print('"', end="", flush=True)
            break

        input_ids_list.append(best_token_id)
        print(model.decode([best_token_id]), end="", flush=True)


def generate_number(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        valid_number_token_ids: list[int]) -> None:
    id_comma = int(vocab[","])
    id_close_brace = int(vocab["}"])
    id_close_bracket = int(vocab["]"])

    token_count = 0
    max_tokens = 15

    while True:
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[valid_number_token_ids] = logits[
            valid_number_token_ids]
        logits_contraints[
            [id_comma, id_close_brace, id_close_bracket]] = logits[
            [id_comma, id_close_brace, id_close_bracket]]
        best_token_id = int(np.argmax(logits_contraints))
        token_count += 1

        if best_token_id in (id_comma, id_close_brace, id_close_bracket) or \
                token_count >= max_tokens:
            break

        input_ids_list.append(best_token_id)
        print(model.decode([best_token_id]), end="", flush=True)


def generate_boolean(
        model: Any, input_ids_list: list[int],
        valid_bool_token_ids: list[int]) -> None:
    logits = np.array(model.get_logits_from_input_ids(input_ids_list))
    logits_contraints = np.full_like(logits, -np.inf)
    logits_contraints[valid_bool_token_ids] = logits[valid_bool_token_ids]
    best_token_id = int(np.argmax(logits_contraints))
    input_ids_list.append(best_token_id)
    print(model.decode([best_token_id]), end="", flush=True)


def generate_null(model: Any, input_ids_list: list[int]) -> None:
    force_sequence(model, input_ids_list, "null")


def generate_list(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        inner_type: type, valid_string_token_ids: list[int],
        valid_number_token_ids: list[int], closing_quote_token_ids: list[int],
        valid_bool_token_ids: list[int]) -> None:
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
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[id_comma_simple] = logits[id_comma_simple]
        logits_contraints[id_comma_space] = logits[id_comma_space]
        logits_contraints[id_close] = logits[id_close] - malus
        best_token_id = int(np.argmax(logits_contraints))
        if best_token_id == id_close or item_count >= max_items:
            force_sequence(model, input_ids_list, "]")
            break
        force_sequence(model, input_ids_list, ", ")


def generate_value(
        model: Any, input_ids_list: list[int], vocab: dict[str, int],
        expected_type: type, valid_number_token_ids: list[int],
        valid_string_token_ids: list[int], closing_quote_token_ids: list[int],
        valid_bool_token_ids: list[int]) -> None:
    origin_type = get_origin(expected_type) or expected_type
    if origin_type in (float, int):
        generate_number(model, input_ids_list, vocab, valid_number_token_ids)
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
