from llm_sdk import Small_LLM_Model
import numpy as np
import json


def main() -> None:
    model = Small_LLM_Model()
    prompt = "What is the sum of 2 and 3? The answer is"
    print(f"Prompt: '{prompt}'")

    input_ids = model.encode(prompt)
    print(f"Input IDs: {input_ids}")

    input_ids_list = input_ids[0].tolist()

    vocab_path = model.get_path_to_vocab_file()
    print(model.get_path_to_vocab_file())
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    eos_token_id = vocab.get("<|endoftext|>")

    # ... (ton code d'encodage ici) ...

    best_token_id = -1

    while best_token_id != eos_token_id:
        logits = model.get_logits_from_input_ids(input_ids_list)
        best_token_id = int(np.argmax(logits))

        if best_token_id != eos_token_id:
            print(model.decode([best_token_id]), end="", flush=True)
            input_ids_list.append(best_token_id)

    print("\nTerminé !")


if __name__ == "__main__":
    main()
