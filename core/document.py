import os
import json
import numpy as np
import PyPDF2

from docx import Document
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    EMBEDDING_MODEL_NAME,
)


def clean_text(text):
    if not text:
        return ""

    text = str(text)

    text = text.replace("\x00", " ")

    text = " ".join(
        text.split()
    )

    return text.strip()


def extract_pdf(path):
    text_parts = []

    try:
        with open(
            path,
            "rb"
        ) as f:

            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:

                try:
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        text_parts.append(page_text)

                except Exception as e:
                    print(
                        f"PDF page extraction error {path}:",
                        e
                    )

    except Exception as e:

        print(
            f"PDF error {path}:",
            e
        )

    return clean_text(
        "\n".join(text_parts)
    )


def extract_docx(path):
    text = ""

    try:

        document = Document(path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(
            paragraphs
        )

    except Exception as e:

        print(
            f"DOCX error {path}:",
            e
        )

    return clean_text(text)


def extract_txt(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return clean_text(
                f.read()
            )

    except Exception as e:

        print(
            f"TXT error {path}:",
            e
        )

        return ""


def extract_document(path):

    if not path:
        return ""

    extension = (
        os.path.splitext(path)[1]
        .lower()
    )

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    if extension in (
        ".txt",
        ".md",
        ".text"
    ):
        return extract_txt(path)

    return ""


def create_chunks(
    text,
    source,
    chunk_size=500,
    overlap=80
):

    words = text.split()

    if not words:
        return []

    result = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        if len(chunk_text) > 30:

            result.append(
                {
                    "source": source,
                    "text": chunk_text
                }
            )

        if end >= len(words):
            break

        start = end - overlap

    return result


def load_embedder(status_callback=None):

    try:

        if status_callback:
            status_callback(
                "Loading local embedding model..."
            )

        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        if status_callback:
            status_callback(
                "Embedding model ready."
            )

        return model

    except Exception as e:

        print(
            "Embedder error:",
            e
        )

        if status_callback:
            status_callback(
                f"Embedder error: {str(e)[:60]}"
            )

        return None


def build_faiss_index(vectors):

    if vectors is None:
        return None

    matrix = np.asarray(
        vectors,
        dtype=np.float32
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
    ):
        return None

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


def load_cached_memory(
    target_folder,
    status_callback=None
):

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
        return None

    try:

        with open(
            chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            cached_chunks = json.load(f)

        cached_embeddings = np.load(
            embeddings_file
        )

        cached_index = None

        if os.path.exists(
            faiss_file
        ):

            try:

                cached_index = faiss.read_index(
                    faiss_file
                )

            except Exception:
                cached_index = None

        if cached_index is None:
            cached_index = build_faiss_index(
                cached_embeddings
            )

        if status_callback:
            status_callback(
                f"Loaded {len(cached_chunks)} cached chunks."
            )

        return (
            cached_chunks,
            cached_embeddings,
            cached_index
        )

    except Exception as e:

        print(
            "Cache loading error:",
            e
        )

        return None
