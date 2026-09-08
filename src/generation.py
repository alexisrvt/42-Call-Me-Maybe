import numpy as np
# import time
# from contextlib import contextmanager


# @contextmanager
# def track_logit_time():
#     start = time.perf_counter()
#     yield
#     elapsed = (time.perf_counter() - start) * 1000
#     print(f" [{elapsed:.1f}ms]", end="", flush=True)


def get_token_masks(vocab: dict) -> tuple[list[int], list[int], list[int]]:
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

    return (
        list(valid_number_token_ids), list(valid_string_token_ids),
        list(closing_quote_token_ids)
    )


def force_sequence(model, input_ids_list: list[int], text: str) -> None:
    target_ids = model.encode(text)[0].tolist()
    for expected_token in target_ids:
        input_ids_list.append(expected_token)
    print(text, end="", flush=True)


def generate_string(
        model, input_ids_list: list[int], vocab: dict[str, int],
        valid_string_token_ids: list[int],
        closing_quote_token_ids: list[int]) -> None:
    id_quote = int(vocab['"'])
    input_ids_list.append(id_quote)
    print('"', end="", flush=True)

    token_count = 0
    max_tokens = 40

    while True:
        # with track_logit_time():
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
        model, input_ids_list: list[int], vocab: dict[str, int],
        valid_number_token_ids: list[int]) -> None:
    id_comma = int(vocab[","])
    id_close_brase = int(vocab["}"])

    token_count = 0
    max_tokens = 15

    while True:
        # with track_logit_time():
        logits = np.array(model.get_logits_from_input_ids(input_ids_list))
        logits_contraints = np.full_like(logits, -np.inf)
        logits_contraints[valid_number_token_ids] = logits[
            valid_number_token_ids]
        logits_contraints[[id_comma, id_close_brase]] = logits[
            [id_comma, id_close_brase]]
        best_token_id = int(np.argmax(logits_contraints))
        token_count += 1

        if best_token_id in (id_comma, id_close_brase) or \
                token_count >= max_tokens:
            break

        input_ids_list.append(best_token_id)
        print(model.decode([best_token_id]), end="", flush=True)
