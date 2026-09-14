*This project has been created as part of the 42 curriculum by arivet.*

## Description

**Call Me Maybe** is a robust function-calling and constrained decoding engine designed for Small Language Models (SLMs). The goal of this project is to force an LLM to generate 100% syntactically valid JSON that strictly adheres to dynamic function signatures. By intercepting the generation process at the token level, the engine translates JSON Schemas into Pydantic models and applies strict vocabulary masking, completely eliminating parsing errors, format hallucinations, and type mismatches.

## Instructions

This project uses `uv` for fast, cross-platform deterministic dependency management. A `Makefile` is provided to automate common tasks.

* **Installation:** Run `make install` to create the virtual environment and install dependencies via `uv sync`.
* **Execution:** Run `make run` to execute the pipeline with the default input files.
* **Debugging:** Run `make debug` to launch the script with Python's built-in `pdb`.
* **Cleaning:** Run `make clean` to remove `__pycache__` and `.mypy_cache` directories.
* **Linting:** Run `make lint` or `make lint-strict` to validate the codebase against `flake8` and strict `mypy` typing rules.

## Example Usage

You can run the program using the Makefile or directly via the CLI, passing custom JSON files for definitions and prompts:

```bash
# Using the default Makefile rule
make run

# Using the CLI with custom arguments
uv run python -m src \
    --input data/input/custom_tests.json \
    --functions_definition data/input/custom_functions.json \
    --output data/output/results.json

```

## Algorithm Explanation

The constrained decoding approach relies on manipulating the model's logits before the sampling phase (using `np.argmax`).

1. **Token Masking:** At initialization, the script pre-calculates valid token IDs from the BPE vocabulary (e.g., isolating digits for numbers, or `true`/`false` for booleans).
2. **Recursive Routing:** A main `generate_value` function evaluates the expected Pydantic type (string, int, list, nested BaseModel) and routes it to the dedicated generation loop.
3. **Logit Constraint:** Inside the generation loops, logits for invalid tokens are set to `-np.inf`. Only syntactically valid tokens (e.g., quotes, commas, brackets, or type-specific characters) are allowed to be predicted.
4. **Forced Sequences:** For static structural elements (like dictionary keys or JSON syntax), the engine bypasses generation entirely and uses `force_sequence` to inject the exact required token IDs, saving computation time.

## Design Decisions

* **Pydantic Integration:** `json-schema-to-pydantic` is used to dynamically convert the parsed JSON schema definitions into strict Python object models at runtime, allowing the recursive generator to iterate over `model_fields`.
* **Cross-Platform Portability:** `uv` is configured with multi-platform environments in `pyproject.toml` to ensure the `uv.lock` file resolves perfectly on the 42 Linux cluster, macOS, and Windows.
* **Dynamic Schema Normalization:** A runtime validation step was added to silently normalize legacy/simplified JSON schemas into standard JSON Schema formats before passing them to Pydantic, ensuring backward compatibility with older test files without modifying the raw data.

## Performance Analysis

* **Speed:** By utilizing NumPy arrays for rapid logit masking and forcing static sequences, the engine processes complex, deeply nested prompts in approximately 1.5 to 2 minutes on standard 42 Roma cluster machines—well below the 5-minute threshold.
* **Accuracy & Reliability:** The strict token masking guarantees absolute JSON validity. The output will never contain unescaped quotes, trailing commas, or incorrect types, guaranteeing 100% safe parsing via `json.loads()`.

## Challenges Faced

* **BPE Tokenization Quirks:** The LLM's tokenizer treats `,` and `, ` (comma with a space) as completely separate tokens. Restricting the model to only the raw `,` caused its probability distribution to collapse, leading to early stopping. **Solution:** Allowed both comma tokens in the logit mask, then forced the standardized `, ` sequence manually to maintain formatting.
* **Early List Closure (The "Lazy LLM" Problem):** Small models heavily favor closing arrays `]` prematurely when generating complex nested objects because it requires less probabilistic effort. **Solution:** Implemented a dynamic logit malus. The probability of the `]` token is penalized by `-10.0` when generating complex Pydantic sub-objects, forcing the model to scan the entire prompt context exhaustively before closing the array.

## Testing Strategy

The implementation was validated using iterative, high-complexity JSON schemas:

* **Deep Nesting & Types:** Tested against schemas requiring nested arrays of objects (e.g., a Pokémon team builder with specific stats objects, or C project memory leak analyzers).
* **Primitive Types:** Validated seamless handling of booleans (`true`/`false` without quotes) and `null` values.
* **Static Typing:** The entire `src/` directory is validated using `mypy --strict`, enforcing proper use of `typing.Any`, generic type arguments, and `typing.cast` for dynamic JSON loading.

## Resources

* **Documentation:** [Pydantic Official Documentation](https://www.google.com/search?q=https://docs.pydantic.dev/) for dynamic model generation and field inspection.
* **AI Usage:** Gemini was used as a thought partner and pair-programming assistant throughout the project. It was primarily utilized to debug BPE tokenization edge cases (identifying the space-comma token discrepancy), to design the mathematical dynamic malus strategy for array generation logits, and to refactor the codebase to satisfy the stringent requirements of `mypy --strict`.