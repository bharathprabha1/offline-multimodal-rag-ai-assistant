from pathlib import Path
import threading

from llama_cpp import Llama


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(
    BASE_DIR
    / "models"
    / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

CONTEXT_SIZE = 1536
GPU_LAYERS = 16
CPU_THREADS = 8
MAX_OUTPUT_TOKENS = 96

_llm = None
_llm_lock = threading.Lock()


def load_llm():
    """
    Load Mistral once and reuse the same local model instance.
    """

    global _llm

    if _llm is not None:
        return _llm

    with _llm_lock:

        if _llm is not None:
            return _llm

        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Mistral model not found: {MODEL_PATH}"
            )

        print("Loading Mistral 7B on GPU...")

        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=CONTEXT_SIZE,
            n_threads=CPU_THREADS,
            n_gpu_layers=GPU_LAYERS,
            verbose=False,
        )

        print("Mistral ready.")

        return _llm


def generate_stream(
    prompt,
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    stop=None
):
    """
    Stream tokens from the persistent local Mistral model.

    Returns the llama.cpp streaming iterator.
    """

    model = load_llm()

    # llama.cpp requires max_tokens to be numeric.
    # Protect against callers accidentally passing a string.
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = MAX_OUTPUT_TOKENS

    # Keep the value valid.
    if max_tokens <= 0:
        max_tokens = MAX_OUTPUT_TOKENS

    if stop is None:
        stop = ["</s>"]

    return model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=stop,
        echo=False,
        stream=True
    )


def generate_answer(
    prompt,
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    stop=None
):
    """
    Non-streaming convenience wrapper.
    """

    stream = generate_stream(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=stop
    )

    parts = []

    for output in stream:

        try:
            token = (
                output
                .get("choices", [{}])[0]
                .get("text", "")
            )
        except Exception:
            token = ""

        if token:
            parts.append(token)

    return "".join(parts).strip()