import re

from .config import (
    OFFLINE_SYSTEM_PROMPT,
    MAX_OUTPUT_TOKENS,
)


# ============================================================
# CONSTANTS
# ============================================================

NO_CONTEXT_ANSWER = (
    "I could not find this information in the provided documents."
)


# ============================================================
# ANSWER STYLE DETECTION
# ============================================================

def detect_answer_style(query):

    q = str(query or "").strip().lower()

    if re.search(
        r"\b(one word|one-word|single word|just the answer|"
        r"only the answer)\b",
        q
    ):
        return "one_word"

    if re.search(
        r"\b(briefly|brief|in short|shortly|short answer|"
        r"small sentence|small explanation|very short|"
        r"keep it short|concise)\b",
        q
    ):
        return "brief"

    if re.search(
        r"\b(elaborate more|elaborate|expand|expand more|"
        r"explain more|more details|in detail|detailed|"
        r"explain in detail|deeply|deep explanation)\b",
        q
    ):
        return "detailed"

    if re.search(
        r"\b(clearly|explain clearly|clear explanation|"
        r"simple explanation|explain simply|simplify|"
        r"easy explanation)\b",
        q
    ):
        return "clear"

    if re.search(
        r"\b(step by step|steps|step-by-step)\b",
        q
    ):
        return "steps"

    if re.search(
        r"\b(summarize|summary|summarise)\b",
        q
    ):
        return "summary"

    return "default"


# ============================================================
# OUTPUT TOKEN LIMIT
# ============================================================

def answer_max_tokens(query):

    style = detect_answer_style(query)

    if style == "one_word":
        return 16

    if style == "brief":
        return 80

    if style == "summary":
        return 140

    if style == "clear":
        return 180

    if style == "steps":
        return 240

    if style == "detailed":
        return 360

    return min(
        260,
        MAX_OUTPUT_TOKENS
    )


# ============================================================
# ANSWER STYLE INSTRUCTION
# ============================================================

def answer_style_instruction(query):

    style = detect_answer_style(query)

    if style == "one_word":
        return """
ANSWER STYLE:

Return only the requested word.

Do not add explanation or extra sentences.
"""

    if style == "brief":
        return """
ANSWER STYLE:

Give a short answer in approximately 1–2 clear sentences.

Do not provide unnecessary background information.
"""

    if style == "detailed":
        return """
ANSWER STYLE:

Provide a detailed but focused explanation.

Include important supporting facts from the supplied context.
Do not repeat information unnecessarily.
"""

    if style == "clear":
        return """
ANSWER STYLE:

Explain the answer using simple and direct language.

Prefer short sentences and explain technical terms when necessary.
Stay strictly within the supplied context.
"""

    if style == "steps":
        return """
ANSWER STYLE:

Present the answer as clear numbered steps when appropriate.

Each step must be supported by the supplied context.
"""

    if style == "summary":
        return """
ANSWER STYLE:

Give only the key points needed to understand the answer.

Keep the response concise and organized.
"""

    return """
ANSWER STYLE:

Give a direct and useful answer.

Normally use approximately 2–5 sentences depending on the
complexity of the question.

For simple factual questions, state the answer clearly and
briefly explain it.

Do not produce unnecessarily long explanations.
"""


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def is_style_only_followup(query):

    q = str(query or "").strip().lower()

    patterns = [
        r"^(briefly|brief)$",
        r"^(in short|shortly|short answer)$",
        r"^(elaborate|elaborate more)$",
        r"^(expand|expand more)$",
        r"^(explain more|more details)$",
        r"^(clearly|explain clearly)$",
        r"^(explain simply|simplify)$",
        r"^(step by step|steps|step-by-step)$",
        r"^(summarize|summary|summarise)$",
        r"^(one word|one-word|single word)$",
        r"^(small sentence|small explanation)$",
    ]

    return any(
        re.fullmatch(pattern, q)
        for pattern in patterns
    )


# ============================================================
# FOLLOW-UP SCOPE
# ============================================================

def build_answer_scope_instruction(
    query,
    style_followup=False,
    previous_topic=""
):

    if style_followup and previous_topic:

        return f"""
ANSWER SCOPE:

This is a follow-up instruction about the previous topic.

Previous topic:
{previous_topic}

The user is changing only the answer style.

Stay focused on the previous topic.

Do not start a new document search.

Do not introduce unrelated information.
"""

    return """
ANSWER SCOPE:

Answer the user's current question directly.

Use ONLY the supplied local document context.

Do not use outside knowledge.
"""


# ============================================================
# DOCUMENT GROUNDING
# ============================================================

def is_document_grounded_question(query):

    q = str(query or "").lower()

    patterns = [
        r"\baccording to\b",
        r"\bin\s+[\w ._-]+\.pdf\b",
        r"\bin\s+[\w ._-]+\.docx\b",
        r"\bin\s+[\w ._-]+\.txt\b",
        r"\bfrom\s+[\w ._-]+\.pdf\b",
        r"\bfrom\s+[\w ._-]+\.docx\b",
        r"\bfrom\s+[\w ._-]+\.txt\b",
        r"\baccording to the document\b",
        r"\baccording to the file\b",
        r"\bin the document\b",
        r"\bin the file\b",
        r"\bthe document says\b",
        r"\bthe file says\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in patterns
    )


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    query,
    context="",
    style_followup=False,
    previous_topic="",
    previous_answer=""
):

    style_instruction = answer_style_instruction(query)

    scope_instruction = build_answer_scope_instruction(
        query,
        style_followup,
        previous_topic
    )

    previous_answer_block = ""

    if previous_answer:
        previous_answer_block = f"""
PREVIOUS ANSWER:

{previous_answer}
"""

    context_block = f"""
LOCAL DOCUMENT CONTEXT:

{context}
"""

    grounding_instruction = f"""
STRICT RAG GROUNDING RULE:

You are answering a question using ONLY the LOCAL DOCUMENT
CONTEXT below.

You MUST NOT use:
- pretrained knowledge;
- general world knowledge;
- assumptions;
- guesses;
- information from the internet;
- information not explicitly supported by the context.

If the answer is not supported by the LOCAL DOCUMENT CONTEXT,
do not attempt to answer it.

Instead return exactly:

{NO_CONTEXT_ANSWER}

Do not add explanations to that fallback response.

Never invent facts.
Never fill missing information from your own knowledge.
Never generate a follow-up question.
Never continue the conversation with a new question.
"""

    return f"""
{OFFLINE_SYSTEM_PROMPT}

{grounding_instruction}

{style_instruction}

{scope_instruction}

{previous_answer_block}

{context_block}

USER QUESTION:

{query}

FINAL ANSWER:

Answer only the user's question using the local context.
"""


# ============================================================
# MODEL ANSWER CLEANING
# ============================================================

def clean_model_answer(answer):

    if not answer:
        return ""

    text = str(answer).strip()

    text = re.sub(
        r"^(assistant|ai)\s*:\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"^final answer\s*:\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"\n*(USER QUESTION|LOCAL DOCUMENT CONTEXT|"
        r"ANSWER STYLE|ANSWER SCOPE|FINAL ANSWER)\s*:.*$",
        "",
        text,
        flags=re.I | re.S
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# PROVENANCE CLEANING
# ============================================================

def strip_provenance_from_answer(answer):

    if not answer:
        return ""

    answer = re.sub(
        r"\s*\[Source:[^\]]+\]",
        "",
        answer,
        flags=re.I
    )

    answer = re.sub(
        r"\s*📄\s*Sources?:.*$",
        "",
        answer,
        flags=re.I | re.S
    )

    return clean_model_answer(answer)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    context="",
    style_followup=False,
    previous_topic="",
    previous_answer=""
):
    """
    Generate a STRICTLY GROUNDED answer.

    If no local context exists, the LLM is NOT called.
    """

    if not query:
        return ""

    # ========================================================
    # CRITICAL RAG SAFETY CHECK
    # ========================================================
    #
    # Never allow the LLM to answer from its pretrained
    # knowledge when retrieval returned no context.
    #
    if not context or not str(context).strip():
        return NO_CONTEXT_ANSWER

    from .llm import generate_answer as llm_generate_answer

    prompt = build_prompt(
        query=query,
        context=context,
        style_followup=style_followup,
        previous_topic=previous_topic,
        previous_answer=previous_answer
    )

    try:

        max_tokens = answer_max_tokens(query)

        answer = llm_generate_answer(
            prompt,
            max_tokens=max_tokens
        )

    except Exception as e:

        return f"Unable to generate answer: {e}"

    answer = clean_model_answer(answer)
    answer = strip_provenance_from_answer(answer)

    # ========================================================
    # REMOVE ACCIDENTAL PROMPT CONTINUATION
    # ========================================================

    answer = answer.split(
        "\nQuestion:",
        1
    )[0]

    answer = answer.split(
        "\nUser Question:",
        1
    )[0]

    answer = answer.split(
        "\nUSER QUESTION:",
        1
    )[0]

    answer = answer.split(
        "\nFINAL ANSWER:",
        1
    )[0]

    answer = answer.strip()

    # ========================================================
    # EMPTY MODEL RESPONSE
    # ========================================================

    if not answer:
        return NO_CONTEXT_ANSWER

    return answer