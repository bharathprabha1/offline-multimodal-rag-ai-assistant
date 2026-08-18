from llama_cpp import Llama
import os
import time

os.environ["LLAMA_CPP_LOG_LEVEL"] = "error"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

print("=" * 65)
print("       RTX 3050 CUDA MISTRAL 7B BENCHMARK")
print("=" * 65)

if not os.path.isfile(MODEL_PATH):
    print("❌ Model not found:")
    print(MODEL_PATH)
    raise SystemExit(1)

print("✅ Model found")
print(f"Path: {MODEL_PATH}")
print()

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading Mistral 7B with GPU offloading...")

start_load = time.perf_counter()

try:
    llm = Llama(
        model_path=MODEL_PATH,

        # Small context for this benchmark
        n_ctx=1024,

        # CPU threads
        n_threads=8,

        # Start with moderate GPU offloading.
        # We will optimize this later for your 4 GB VRAM.
        n_gpu_layers=20,

        verbose=True
    )
except Exception as e:
    print()
    print("❌ MODEL LOAD FAILED")
    print(e)
    raise SystemExit(1)

load_time = time.perf_counter() - start_load

print()
print("=" * 65)
print("✅ MODEL LOADED")
print("=" * 65)
print(f"Load time: {load_time:.2f} seconds")
print()

# ------------------------------------------------------------
# Inference benchmark
# ------------------------------------------------------------

prompt = """You are a local offline AI assistant.

Answer this question in one short sentence:

What is artificial intelligence?

Answer:"""

print("Running inference...")
print()

start_generation = time.perf_counter()

try:
    response = llm(
        prompt,
        max_tokens=40,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>", "Question:"],
        echo=False
    )

    elapsed = time.perf_counter() - start_generation

    answer = response["choices"][0]["text"].strip()

    usage = response.get("usage", {})
    tokens = usage.get("completion_tokens", 0)

    print("🤖 RESPONSE")
    print("-" * 65)
    print(answer)
    print("-" * 65)

    print()
    print(f"Generation time : {elapsed:.2f} seconds")
    print(f"Tokens generated: {tokens}")

    if tokens:
        speed = tokens / elapsed
        print(f"Generation speed: {speed:.2f} tokens/sec")

    print()
    print("=" * 65)
    print("✅ GPU BENCHMARK COMPLETE")
    print("=" * 65)

except Exception as e:
    print()
    print("❌ INFERENCE FAILED")
    print(e)

finally:
    try:
        del llm
    except Exception:
        pass