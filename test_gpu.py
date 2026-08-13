from llama_cpp import Llama
import os
import time

# ============================================================
# LOCAL LLAMA.CPP PERFORMANCE TEST
# Designed for:
#   Intel i5-12450H
#   16 GB RAM
#   RTX 3050 4 GB
# ============================================================

os.environ["LLAMA_CPP_LOG_LEVEL"] = "error"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

print("=" * 60)
print("       LOCAL MISTRAL 7B PERFORMANCE TEST")
print("=" * 60)

print(f"Model: {MODEL_PATH}")

if not os.path.isfile(MODEL_PATH):
    print()
    print("❌ MODEL NOT FOUND")
    print(MODEL_PATH)
    raise SystemExit(1)

print("✅ Model file found")
print()

try:
    print("Loading model...")
    start_load = time.time()

    llm = Llama(
        model_path=MODEL_PATH,

        # Keep memory requirements reasonable
        n_ctx=1024,

        # Your CPU has 12 logical threads.
        # Start conservatively.
        n_threads=8,

        # Current installation is CPU based.
        n_gpu_layers=0,

        verbose=False
    )

    load_time = time.time() - start_load

    print()
    print("✅ MODEL LOADED")
    print(f"Load time: {load_time:.2f} seconds")
    print()

except Exception as e:
    print()
    print("❌ MODEL LOADING FAILED")
    print(e)
    raise SystemExit(1)


# ============================================================
# VERY SMALL GENERATION TEST
# ============================================================

print("=" * 60)
print("STARTING INFERENCE TEST")
print("=" * 60)

prompt = """You are a local offline AI assistant.
Answer in one short sentence.

Question: What is artificial intelligence?

Answer:"""

try:
    start_generation = time.time()

    response = llm(
        prompt,
        max_tokens=20,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>", "Question:"],
        echo=False
    )

    generation_time = time.time() - start_generation

    answer = response["choices"][0]["text"].strip()

    usage = response.get("usage", {})

    completion_tokens = usage.get("completion_tokens", 0)

    print()
    print("🤖 AI RESPONSE")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    print()
    print(f"Generation time : {generation_time:.2f} seconds")
    print(f"Tokens generated: {completion_tokens}")

    if completion_tokens > 0:
        speed = completion_tokens / generation_time
        print(f"Generation speed: {speed:.2f} tokens/sec")

    print()
    print("✅ INFERENCE TEST PASSED")

except KeyboardInterrupt:
    print()
    print("⚠️ Generation was interrupted.")
    print("The model loaded successfully, but generation was too slow.")

except Exception as e:
    print()
    print("❌ INFERENCE FAILED")
    print(e)

finally:
    try:
        del llm
    except Exception:
        pass

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)