import threading
import rag
from core.ingestion import search_documents


FILES_FOLDER = r".\files"


# ============================================================
# 1. INGEST DOCUMENTS
# ============================================================

print("=" * 70)
print("OFFLINE MULTIMODAL RAG - FULL RETRIEVAL TEST")
print("=" * 70)

print("\n[1] Starting ingestion...\n")

ok = rag.run_ingestion(
    FILES_FOLDER,
    lambda msg: print("[STATUS]", msg),
    threading.Event()
)

print("\n" + "=" * 70)

if not ok:
    print("❌ INGESTION FAILED")
    raise SystemExit(1)

print("✅ INGESTION SUCCESSFUL")
print("Chunks      :", len(rag.chunks))
print(
    "Embeddings  :",
    None if rag.embeddings is None else rag.embeddings.shape
)
print(
    "FAISS vectors:",
    None if rag.faiss_index is None else rag.faiss_index.ntotal
)

print("=" * 70)


# ============================================================
# 2. TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "What is the capital of India?",
    "What is the capital of Germany?",
    "What is the capital of China?",
    "What information is available about student scores?",
    "What database concepts are mentioned in the documents?",
    "What is the population of Mars according to the documents?",
]


# ============================================================
# 3. RETRIEVAL TEST
# ============================================================

for number, question in enumerate(TEST_QUESTIONS, start=1):

    print("\n")
    print("=" * 70)
    print(f"TEST {number}")
    print("=" * 70)

    print("QUESTION:")
    print(question)

    print("\nSearching FAISS...\n")

    try:

        results = search_documents(
            question,
            embedder=rag.embedder,
            state=rag.CORE_STATE,
            top_k=5
        )

    except Exception as e:

        print("❌ SEARCH ERROR:")
        print(e)
        continue


    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        print("❌ NO RESULTS")
        continue


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(f"✅ Retrieved {len(results)} results\n")

    for result in results:

        rank = result.get("rank", "?")
        source = result.get("source", "UNKNOWN")
        score = result.get("score", 0)
        text = result.get("text", "")

        print("-" * 70)

        print(f"Rank   : {rank}")
        print(f"Source : {source}")
        print(f"Score  : {score:.4f}")

        print("\nText:")
        print(text[:500])

        print()


# ============================================================
# 4. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVAL TEST COMPLETE")
print("=" * 70)

print("Documents ingested :", len(rag.chunks))
print(
    "FAISS vectors      :",
    None if rag.faiss_index is None else rag.faiss_index.ntotal
)

print("\nNext step:")
print("Verify that the correct source appears at Rank 1.")
print("=" * 70)