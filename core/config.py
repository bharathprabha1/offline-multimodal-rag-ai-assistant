import os

# ============================================================
# OFFLINE ENVIRONMENT
# ============================================================

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LLAMA_CPP_LOG_LEVEL", "error")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILENAME = (
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    MODEL_FILENAME
)

VOSK_PATH = os.path.join(
    BASE_DIR,
    "vosk-model-small-en-us-0.15"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "chat_history.json"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# GPU SETTINGS
# DO NOT CHANGE — CURRENT WORKING CONFIGURATION
# ============================================================

GPU_LAYERS = 16

CONTEXT_SIZE = 1536

CPU_THREADS = 8

MAX_OUTPUT_TOKENS = 512


# ============================================================
# RAG SETTINGS
# CURRENT VALUES PRESERVED
# ============================================================

RAG_MAX_CONTEXT_CHUNKS = 3

RAG_MAX_CHUNK_CHARS = 1800

RAG_MAX_CONTEXT_CHARS = 4200

RAG_MIN_GENERATION_TOKENS = 96

RAG_CONTEXT_RESERVE_TOKENS = 32

TOP_K = 4

SIMILARITY_THRESHOLD = 0.25


# ============================================================
# OFFLINE SYSTEM PROMPT
# ============================================================

OFFLINE_SYSTEM_PROMPT = """
You are an offline local AI assistant running entirely on the user's computer.

IMPORTANT RULES:

1. You do NOT have internet access.
2. You cannot browse websites.
3. You cannot access online databases.
4. You cannot search Google, Bing, Wikipedia, or any website.
5. Never claim that you can access the internet.
6. Never pretend that you performed an online search.
7. If the user asks for current online information, clearly explain that
   internet access is unavailable.
8. You can use information from:
   - the local language model,
   - documents provided by the user,
   - retrieved local RAG context,
   - the current conversation.
9. If local documents do not contain the requested information, say that
   the information was not found in the provided local documents instead
   of inventing a document-based answer.
10. Be concise, factual and helpful.
"""
