import sys
import os
import json

import numpy as np
import faiss

from .config import (
    RAG_MAX_CONTEXT_CHUNKS,
    RAG_MAX_CHUNK_CHARS,
    RAG_MAX_CONTEXT_CHARS,
    TOP_K,
    SIMILARITY_THRESHOLD,
)


# ============================================================
# SHARED RAG STATE
# ============================================================
#
# These variables are intentionally exposed at module level.
# The GUI and retrieval layer can access them directly through:
#
#     core.ingestion.chunks
#     core.ingestion.embeddings
#     core.ingestion.faiss_index
#     core.ingestion.embedder
#
# The state dictionary used by rag.py is synchronized with these
# values after ingestion/cache loading.
# ============================================================

chunks = []
embeddings = None
faiss_index = None
embedder = None


# ============================================================
# STATE SYNCHRONIZATION
# ============================================================

def _sync_module_state(state):
    """
    Synchronize the module-level RAG state with a state dictionary.
    """

    global chunks
    global embeddings
    global faiss_index
    global embedder

    if state is None:
        return

    chunks = state.get(
        "chunks",
        []
    )

    embeddings = state.get(
        "embeddings"
    )

    faiss_index = state.get(
        "faiss_index"
    )

    embedder = state.get(
        "embedder"
    )


def _sync_state_dict(state):
    """
    Synchronize a state dictionary with the module-level RAG state.
    """

    if state is None:
        return

    state["chunks"] = chunks
    state["embeddings"] = embeddings
    state["faiss_index"] = faiss_index
    state["embedder"] = embedder


# ============================================================
# EMBEDDER COMPATIBILITY
# ============================================================

def load_embedder(status_callback=None):
    """
    Load the SentenceTransformer embedding model.

    The actual model loader lives in core.document.
    """

    global embedder

    try:

        from .document import load_embedder as document_load_embedder

        result = document_load_embedder(
            status_callback=status_callback
        )

        # The document module may return the actual model,
        # True/False, or use its own module-level variable.

        if (
            result is not None
            and result is not True
            and result is not False
        ):

            embedder = result

        else:

            try:

                from . import document

                document_embedder = getattr(
                    document,
                    "embedder",
                    None
                )

                if document_embedder is not None:

                    embedder = document_embedder

            except Exception:
                pass

        if (
            embedder is None
            and callable(
                getattr(
                    document_load_embedder,
                    "encode",
                    None
                )
            )
        ):

            embedder = document_load_embedder

        if (
            embedder is not None
            and callable(
                getattr(
                    embedder,
                    "encode",
                    None
                )
            )
        ):

            return embedder

        return result

    except Exception as e:

        if status_callback:
            status_callback(
                f"Embedding model error: {e}"
            )
        else:
            print(
                "Embedding model error:",
                e
            )

        return None


# ============================================================
# FAISS
# ============================================================

def build_faiss_index(vectors):
    """
    Build a FAISS inner-product index from embedding vectors.

    Vectors are L2-normalized first, so inner product is
    equivalent to cosine similarity.
    """

    if vectors is None:
        return None

    matrix = np.asarray(
        vectors,
        dtype=np.float32
    )

    if matrix.ndim != 2:
        return None

    if matrix.shape[0] == 0:
        return None

    if matrix.shape[1] == 0:
        return None

    # Normalize a copy so the caller's original array is not
    # unexpectedly modified.
    matrix = matrix.copy()

    faiss.normalize_L2(
        matrix
    )

    dimension = int(
        matrix.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        matrix
    )

    return index


# ============================================================
# LOAD CACHED MEMORY
# ============================================================

def load_cached_memory(
    target_folder,
    status_callback=None,
    state=None
):
    """
    Load previously generated chunks, embeddings and FAISS index.

    state must contain:
        chunks
        embeddings
        faiss_index
    """

    global chunks
    global embeddings
    global faiss_index

    if state is None:
        raise ValueError(
            "load_cached_memory requires a state dictionary."
        )

    cache_dir = os.path.join(
        target_folder,
        "cache"
    )

    chunks_file = os.path.join(
        cache_dir,
        "chunks.json"
    )

    embeddings_file = os.path.join(
        cache_dir,
        "embeddings.npy"
    )

    faiss_file = os.path.join(
        cache_dir,
        "faiss.index"
    )

    if not (
        os.path.exists(chunks_file)
        and os.path.exists(embeddings_file)
    ):
        return False

    try:

        # ----------------------------------------------------
        # LOAD CHUNKS
        # ----------------------------------------------------

        with open(
            chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            cached_chunks = json.load(f)

        if not isinstance(
            cached_chunks,
            list
        ):

            print(
                "Invalid cached chunks format."
            )

            return False

        # ----------------------------------------------------
        # LOAD EMBEDDINGS
        # ----------------------------------------------------

        cached_embeddings = np.load(
            embeddings_file
        )

        cached_embeddings = np.asarray(
            cached_embeddings,
            dtype=np.float32
        )

        if cached_embeddings.ndim != 2:

            print(
                "Invalid cached embeddings shape."
            )

            return False

        if (
            len(cached_chunks)
            != cached_embeddings.shape[0]
        ):

            print(
                "Cache size mismatch."
            )

            return False

        if len(cached_chunks) == 0:

            print(
                "Cached memory is empty."
            )

            return False

        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state["chunks"] = cached_chunks

        state["embeddings"] = cached_embeddings

        # ----------------------------------------------------
        # LOAD FAISS
        # ----------------------------------------------------

        loaded_index = None

        if os.path.exists(
            faiss_file
        ):

            try:

                loaded_index = faiss.read_index(
                    faiss_file
                )

                if (
                    loaded_index.ntotal
                    != len(cached_chunks)
                ):

                    print(
                        "FAISS cache size mismatch. "
                        "Rebuilding index."
                    )

                    loaded_index = None

                elif (
                    loaded_index.d
                    != cached_embeddings.shape[1]
                ):

                    print(
                        "FAISS dimension mismatch. "
                        "Rebuilding index."
                    )

                    loaded_index = None

            except Exception as e:

                print(
                    "FAISS cache load error:",
                    e
                )

                loaded_index = None

        # ----------------------------------------------------
        # REBUILD FAISS IF REQUIRED
        # ----------------------------------------------------

        if loaded_index is None:

            loaded_index = build_faiss_index(
                cached_embeddings
            )

            if loaded_index is None:
                return False

            try:

                os.makedirs(
                    cache_dir,
                    exist_ok=True
                )

                faiss.write_index(
                    loaded_index,
                    faiss_file
                )

            except Exception as e:

                print(
                    "FAISS cache save warning:",
                    e
                )

        state["faiss_index"] = loaded_index

        # ----------------------------------------------------
        # SYNCHRONIZE MODULE STATE
        # ----------------------------------------------------

        _sync_module_state(
            state
        )

        if status_callback:

            status_callback(
                f"Loaded FAISS memory: "
                f"{len(chunks)} segments."
            )

        return True

    except Exception as e:

        print(
            "Cache load error:",
            e
        )

        state["faiss_index"] = None

        _sync_module_state(
            state
        )

        return False


# ============================================================
# INGESTION
# ============================================================

def run_ingestion(
    target_folder,
    status_callback,
    stop_event=None,
    state=None,
    load_embedder=None,
    extract_pdf=None,
    extract_docx=None,
    extract_txt=None,
    process_audio=None,
    create_chunks=None,
    embedder=None
):
    """
    Modular document ingestion pipeline.

    Pipeline:

        Folder
          ↓
        Document extraction
          ↓
        Chunking
          ↓
        SentenceTransformer embeddings
          ↓
        FAISS index
          ↓
        Shared RAG state
    """

    global chunks
    global embeddings
    global faiss_index

    if state is None:
        raise ValueError(
            "run_ingestion requires a state dictionary."
        )

    if extract_pdf is None:
        raise ValueError(
            "run_ingestion requires extract_pdf."
        )

    if extract_docx is None:
        raise ValueError(
            "run_ingestion requires extract_docx."
        )

    if extract_txt is None:
        raise ValueError(
            "run_ingestion requires extract_txt."
        )

    if process_audio is None:
        raise ValueError(
            "run_ingestion requires process_audio."
        )

    if create_chunks is None:
        raise ValueError(
            "run_ingestion requires create_chunks."
        )

    # --------------------------------------------------------
    # EMBEDDER
    # --------------------------------------------------------

    if (
        embedder is None
        or not callable(
            getattr(
                embedder,
                "encode",
                None
            )
        )
    ):

        if load_embedder is None:
            raise ValueError(
                "No valid embedder and no load_embedder function."
            )

        loaded_embedder = load_embedder(
            status_callback
        )

        if (
            loaded_embedder is not None
            and callable(
                getattr(
                    loaded_embedder,
                    "encode",
                    None
                )
            )
        ):

            embedder = loaded_embedder

    if (
        embedder is None
        or not callable(
            getattr(
                embedder,
                "encode",
                None
            )
        )
    ):

        if status_callback:
            status_callback(
                "Invalid embedding model: encode() is unavailable."
            )

        return False

    state["embedder"] = embedder

    # --------------------------------------------------------
    # FOLDER VALIDATION
    # --------------------------------------------------------

    if not target_folder:

        if status_callback:
            status_callback(
                "No folder selected."
            )

        return False

    if not os.path.isdir(
        target_folder
    ):

        if status_callback:
            status_callback(
                f"Folder not found: {target_folder}"
            )

        return False

    if status_callback:
        status_callback(
            f"Scanning folder: "
            f"{os.path.basename(target_folder)}"
        )

    try:

        files = os.listdir(
            target_folder
        )

    except Exception as e:

        if status_callback:
            status_callback(
                f"Folder access error: {e}"
            )

        return False

    if not files:

        if status_callback:
            status_callback(
                "Folder is empty."
            )

        return False

    # --------------------------------------------------------
    # EXTRACT DOCUMENTS
    # --------------------------------------------------------

    all_chunks = []

    total_files = len(
        files
    )

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".wav",
    }

    for index, filename in enumerate(
        files,
        start=1
    ):

        if (
            stop_event is not None
            and stop_event.is_set()
        ):

            if status_callback:
                status_callback(
                    "Document processing stopped."
                )

            return False

        path = os.path.join(
            target_folder,
            filename
        )

        if not os.path.isfile(
            path
        ):
            continue

        extension = (
            os.path.splitext(
                filename
            )[1]
            .lower()
        )

        if extension not in supported_extensions:
            continue

        if status_callback:

            status_callback(
                f"[{index}/{total_files}] "
                f"Reading: {filename}"
            )

        text = ""

        try:

            if extension == ".pdf":

                text = extract_pdf(
                    path
                )

            elif extension == ".docx":

                text = extract_docx(
                    path
                )

            elif extension == ".txt":

                text = extract_txt(
                    path
                )

            elif extension == ".wav":

                text = process_audio(
                    path
                )

        except Exception as e:

            if status_callback:

                status_callback(
                    f"Error reading {filename}: "
                    f"{str(e)[:150]}"
                )

            continue

        if not text:
            continue

        # ----------------------------------------------------
        # CREATE CHUNKS
        # ----------------------------------------------------

        try:

            file_chunks = create_chunks(
                text,
                filename
            )

        except TypeError:

            # Compatibility with older create_chunks(text)
            # implementations.
            try:

                file_chunks = create_chunks(
                    text
                )

            except Exception as e:

                if status_callback:

                    status_callback(
                        f"Chunking error: "
                        f"{str(e)[:150]}"
                    )

                continue

        except Exception as e:

            if status_callback:

                status_callback(
                    f"Chunking error: "
                    f"{str(e)[:150]}"
                )

            continue

        if file_chunks:

            all_chunks.extend(
                file_chunks
            )

    # --------------------------------------------------------
    # CHECK CHUNKS
    # --------------------------------------------------------

    if not all_chunks:

        if status_callback:
            status_callback(
                "No supported document content found."
            )

        return False

    if (
        stop_event is not None
        and stop_event.is_set()
    ):

        if status_callback:
            status_callback(
                "Document processing stopped."
            )

        return False

    if status_callback:

        status_callback(
            f"Creating embeddings for "
            f"{len(all_chunks)} segments..."
        )

    # --------------------------------------------------------
    # PREPARE TEXT
    # --------------------------------------------------------

    texts = []

    for item in all_chunks:

        if isinstance(
            item,
            dict
        ):

            text_value = str(
                item.get(
                    "text",
                    ""
                )
            )

        else:

            text_value = str(
                item
            )

        text_value = text_value.strip()

        if text_value:
            texts.append(
                text_value
            )

        else:
            texts.append(
                ""
            )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    try:

        vectors = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        if vectors.ndim == 1:

            vectors = vectors.reshape(
                1,
                -1
            )

        if vectors.ndim != 2:

            if status_callback:
                status_callback(
                    "Embedding generation returned invalid vectors."
                )

            return False

        if vectors.shape[0] != len(
            all_chunks
        ):

            if status_callback:
                status_callback(
                    "Embedding/chunk count mismatch."
                )

            return False

        if vectors.shape[0] == 0:

            if status_callback:
                status_callback(
                    "Embedding generation returned no vectors."
                )

            return False

    except Exception as e:

        if status_callback:

            status_callback(
                f"Embedding error: {e}"
            )

        return False

    # --------------------------------------------------------
    # STOP CHECK
    # --------------------------------------------------------

    if (
        stop_event is not None
        and stop_event.is_set()
    ):

        if status_callback:
            status_callback(
                "Embedding processing stopped."
            )

        return False

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    try:

        index = build_faiss_index(
            vectors
        )

        if index is None:

            if status_callback:
                status_callback(
                    "Failed to build FAISS index."
                )

            return False

    except Exception as e:

        if status_callback:

            status_callback(
                f"FAISS error: {e}"
            )

        return False

    # --------------------------------------------------------
    # UPDATE SHARED STATE
    # --------------------------------------------------------

    state["chunks"] = all_chunks
    state["embeddings"] = vectors
    state["faiss_index"] = index
    state["embedder"] = embedder

    _sync_module_state(
        state
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if status_callback:

        status_callback(
            f"Ingestion complete: "
            f"{len(chunks)} chunks, "
            f"{vectors.shape[0]} embeddings, "
            f"{faiss_index.ntotal} FAISS vectors."
        )

    return True


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(
    query,
    embedder=None,
    state=None,
    top_k=5,
    min_score=0.40
):
    """
    Retrieve relevant document chunks using FAISS.
    """

    if not query:
        return []

    # --------------------------------------------------------
    # Determine state
    # --------------------------------------------------------

    if state is None:

        # Use module-level state when no explicit state
        # dictionary is supplied.
        local_chunks = chunks
        local_index = faiss_index

        if embedder is None:
            embedder = globals().get(
                "embedder"
            )

    else:

        local_chunks = state.get(
            "chunks",
            []
        )

        local_index = state.get(
            "faiss_index"
        )

        if embedder is None:

            embedder = state.get(
                "embedder"
            )

    if not local_chunks:
        return []

    if local_index is None:
        return []

    if embedder is None:

        raise ValueError(
            "Embedding model is not available."
        )

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    if query_vector.ndim == 1:

        query_vector = query_vector.reshape(
            1,
            -1
        )

    faiss.normalize_L2(
        query_vector
    )

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    search_count = min(
        max(
            int(top_k),
            1
        ),
        len(local_chunks)
    )

    scores, indices = local_index.search(
        query_vector,
        search_count
    )

    results = []

    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        idx = int(
            idx
        )

        if idx < 0:
            continue

        if idx >= len(
            local_chunks
        ):
            continue

        score = float(
            score
        )

        if score < float(
            min_score
        ):
            continue

        item = local_chunks[
            idx
        ]

        if isinstance(
            item,
            dict
        ):

            text_value = str(
                item.get(
                    "text",
                    ""
                )
            )

            source = str(
                item.get(
                    "source",
                    item.get(
                        "filename",
                        "Unknown source"
                    )
                )
            )

        else:

            text_value = str(
                item
            )

            source = "Unknown source"

        if not text_value.strip():
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "source": source,
                "text": text_value,
                "score": score,
            }
        )

    return results