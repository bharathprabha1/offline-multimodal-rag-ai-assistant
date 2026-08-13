from llama_cpp import Llama
import os

# Suppress junk logs
os.environ["LLAMA_CPP_LOG_LEVEL"] = "error"

print("--- GPU TEST STARTING ---")

try:
    # Attempt to load model on GPU
    # NOTE: Make sure the path matches your actual file location!
    llm = Llama(
        model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=-1, # <--- The magic number
        verbose=True
    )
    print("✅ SUCCESS! Model loaded.")

except Exception as e:
    print(f"❌ FAIL: {e}")