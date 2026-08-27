import re
import numpy as np
import faiss

from .config import TOP_K, SIMILARITY_THRESHOLD
from . import ingestion


def normalize_source_name(value):
    if not value:
        return ""

    value = str(value).strip()
    value = value.replace("\\", "/")

    return value.split("/")[-1].strip()


def find_requested_sources(query):

    if not query:
        return []

    q = query.lower()
    found = []

    for chunk in ingestion.chunks:

        source = normalize_source_name(
            chunk.get("source", "")
        )

        if not source:
            continue

        if source.lower() in q:
            if source not in found:
                found.append(source)

    return found


def retrieve_context(
    query,
    top_k=TOP_K,
    similarity_threshold=SIMILARITY_THRESHOLD
):

    if not query:
        return [], ""

    if (
        ingestion.embedder is None
        or ingestion.faiss_index is None
        or not ingestion.chunks
    ):
        return [], ""

    try:

        query_vector = ingestion.embedder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        faiss.normalize_L2(
            query_vector
        )

        search_k = min(
            max(top_k * 3, top_k),
            len(ingestion.chunks)
        )

        scores, indices = ingestion.faiss_index.search(
            query_vector,
            search_k
        )

        requested_sources = find_requested_sources(
            query
        )

        selected = []

        # --------------------------------------------------------
        # EXPLICIT DOCUMENT REQUEST
        # --------------------------------------------------------
        #
        # Example:
        # "According to China.pdf, what is China's capital?"
        #
        # In this situation we should NOT allow another document
        # to become the source simply because it has a higher score.
        #
        if requested_sources:

            for score, index in zip(
                scores[0],
                indices[0]
            ):

                if index < 0:
                    continue

                if index >= len(ingestion.chunks):
                    continue

                item = ingestion.chunks[index]

                source = normalize_source_name(
                    item.get("source", "")
                )

                if source not in requested_sources:
                    continue

                score = float(score)

                if score < similarity_threshold:
                    continue

                selected.append(
                    (
                        index,
                        score
                    )
                )

                if len(selected) >= top_k:
                    break

            if not selected:
                return [], ""

        # --------------------------------------------------------
        # NORMAL QUESTION
        # --------------------------------------------------------
        else:

            for score, index in zip(
                scores[0],
                indices[0]
            ):

                if index < 0:
                    continue

                if index >= len(ingestion.chunks):
                    continue

                score = float(score)

                if score < similarity_threshold:
                    continue

                selected.append(
                    (
                        index,
                        score
                    )
                )

                if len(selected) >= top_k:
                    break

        # --------------------------------------------------------
        # BUILD CONTEXT
        # --------------------------------------------------------

        context_parts = []

        for index, score in selected:

            item = ingestion.chunks[index]

            source = normalize_source_name(
                item.get(
                    "source",
                    ""
                )
            )

            text = item.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            context_parts.append(
                "[Source: "
                + source
                + " | Relevance: "
                + f"{score:.3f}"
                + "]\n"
                + text
            )

        context = "\n\n".join(
            context_parts
        )

        return selected, context

    except Exception as e:

        print(
            "Retrieval error:",
            e
        )

        return [], ""


def get_retrieved_sources(selected):

    sources = []

    for index, score in selected or []:

        try:

            source = normalize_source_name(
                ingestion.chunks[index].get(
                    "source",
                    ""
                )
            )

            if source and source not in sources:

                sources.append(
                    source
                )

        except Exception:
            continue

    return sources


def remove_relevance_metadata(text):

    if not text:
        return ""

    return re.sub(
        r"\s*\|\s*Relevance:\s*[0-9.]+",
        "",
        text,
        flags=re.I
    )
