"""
Sovereign On-Premise Agent
Phase 1: Task Router + Planner

Local-only deterministic task classification.
No internet or external API calls.
"""

from dataclasses import dataclass, field
from typing import List


# ============================================================
# TASK TYPES
# ============================================================

TASK_RAG = "DOCUMENT_QA"
TASK_SUMMARY = "DOCUMENT_SUMMARY"
TASK_IMAGE = "MULTIMODAL_ANALYSIS"
TASK_CODE = "CODE_TASK"
TASK_FILE = "FILE_OPERATION"
TASK_CALCULATION = "CALCULATION"
TASK_GENERAL = "GENERAL"


# ============================================================
# PLAN DATA STRUCTURES
# ============================================================

@dataclass
class PlanStep:
    step_number: int
    action: str
    tool: str
    description: str


@dataclass
class AgentPlan:
    task_type: str
    objective: str
    steps: List[PlanStep] = field(default_factory=list)


# ============================================================
# TASK ROUTER
# ============================================================

class TaskRouter:

    def classify(self, query: str) -> str:

        if not query:
            return TASK_GENERAL

        text = query.lower().strip()

        # ----------------------------------------------------
        # 1. EXPLICIT FILE CREATION / DELIVERABLE REQUESTS
        # ----------------------------------------------------
        file_patterns = [
            "create a word",
            "create word",
            "word document",
            "create an excel",
            "create excel",
            "excel file",
            "spreadsheet",
            "create a powerpoint",
            "create powerpoint",
            "powerpoint",
            "ppt file",
            "pptx file",
            "create a report",
            "create report",
            "generate a report",
            "generate report",
            "approval note",
            "create an approval note",
            "generate an approval note",
        ]

        if any(pattern in text for pattern in file_patterns):
            return TASK_FILE

        # ----------------------------------------------------
        # 2. CODE REQUESTS
        # ----------------------------------------------------
        code_patterns = [
            "write code",
            "write a program",
            "create code",
            "generate code",
            "python code",
            "java code",
            "javascript code",
            "c++ code",
            "program in python",
            "program in java",
            "debug this",
            "debug the code",
            "fix this code",
            "fix the code",
            "write a function",
            "create a function",
            "generate a function",
            "coding task",
            "programming task",
        ]

        if any(pattern in text for pattern in code_patterns):
            return TASK_CODE

        # ----------------------------------------------------
        # 3. MULTIMODAL / IMAGE / SCANNED DOCUMENT REQUESTS
        # ----------------------------------------------------
        image_patterns = [
            "analyze this image",
            "analyse this image",
            "analyze the image",
            "analyse the image",
            "analyze this photo",
            "analyze the photo",
            "analyze this picture",
            "analyze the picture",
            "scanned image",
            "scanned document",
            "scan this",
            "analyze this scan",
            "analyse this scan",
            "p&id",
            "pid drawing",
            "engineering drawing",
            "technical drawing",
            "handwritten note",
            "handwritten document",
            "inspect this image",
            "read this image",
            "read the image",
        ]

        if any(pattern in text for pattern in image_patterns):
            return TASK_IMAGE

        # ----------------------------------------------------
        # 4. SUMMARY REQUESTS
        #
        # Check summary BEFORE generic calculation/question
        # patterns. This prevents "report" or other document
        # words from changing the intended task.
        # ----------------------------------------------------
        summary_patterns = [
            "summarize",
            "summarise",
            "summary",
            "summarize the",
            "summarise the",
            "give a summary",
            "give me a summary",
            "provide a summary",
            "make a summary",
            "key points",
            "main points",
            "important points",
            "brief summary",
            "brief overview",
            "give an overview",
            "give me an overview",
            "provide an overview",
        ]

        if any(pattern in text for pattern in summary_patterns):
            return TASK_SUMMARY

        # ----------------------------------------------------
        # 5. CALCULATION REQUESTS
        # ----------------------------------------------------
        calculation_patterns = [
            "calculate ",
            "calculate",
            "compute ",
            "compute",
            "solve ",
            "solve",
            "percentage",
            "percent",
            "average",
            "mean",
            "sum ",
            "total ",
            "equation",
            "formula",
            "mathematical",
            "mathematics",
        ]

        if any(pattern in text for pattern in calculation_patterns):
            return TASK_CALCULATION

        # ----------------------------------------------------
        # 6. DOCUMENT QUESTIONS / RAG
        # ----------------------------------------------------
        rag_patterns = [
            "according to the document",
            "according to the documents",
            "according to this document",
            "according to these documents",
            "from the document",
            "from the documents",
            "based on the document",
            "based on the documents",
            "in the document",
            "in the documents",
            "what is",
            "what are",
            "who is",
            "who was",
            "when was",
            "when is",
            "where is",
            "where was",
            "why is",
            "why was",
            "how does",
            "how is",
            "how are",
            "explain",
            "tell me about",
        ]

        if any(pattern in text for pattern in rag_patterns):
            return TASK_RAG

        # ----------------------------------------------------
        # 7. GENERAL
        # ----------------------------------------------------
        return TASK_GENERAL

    # ========================================================
    # PLANNER
    # ========================================================

    def create_plan(self, query: str) -> AgentPlan:

        task_type = self.classify(query)

        if task_type == TASK_RAG:

            steps = [
                PlanStep(
                    1,
                    "Retrieve relevant local knowledge",
                    "FAISS",
                    "Search the local knowledge base for relevant document chunks.",
                ),
                PlanStep(
                    2,
                    "Generate grounded answer",
                    "Local Mistral 7B",
                    "Generate an answer using only the retrieved local context.",
                ),
                PlanStep(
                    3,
                    "Return sources",
                    "Source Tracker",
                    "Attach the local source files used for the response.",
                ),
            ]

        elif task_type == TASK_SUMMARY:

            steps = [
                PlanStep(
                    1,
                    "Retrieve relevant document content",
                    "FAISS",
                    "Find the relevant local document segments.",
                ),
                PlanStep(
                    2,
                    "Summarize document content",
                    "Local Mistral 7B",
                    "Create a concise summary grounded in the retrieved content.",
                ),
                PlanStep(
                    3,
                    "Attach sources",
                    "Source Tracker",
                    "Identify the documents supporting the summary.",
                ),
            ]

        elif task_type == TASK_IMAGE:

            steps = [
                PlanStep(
                    1,
                    "Read local image or scanned document",
                    "OCR / Vision",
                    "Extract text and visual information locally.",
                ),
                PlanStep(
                    2,
                    "Retrieve supporting knowledge",
                    "FAISS",
                    "Search the local knowledge base for relevant information.",
                ),
                PlanStep(
                    3,
                    "Analyze multimodal content",
                    "Local Vision / Mistral",
                    "Combine extracted information with local knowledge.",
                ),
            ]

        elif task_type == TASK_CODE:

            steps = [
                PlanStep(
                    1,
                    "Understand coding requirement",
                    "Local Mistral 7B",
                    "Analyze the requested programming task.",
                ),
                PlanStep(
                    2,
                    "Generate code",
                    "Local Mistral 7B",
                    "Generate the requested code locally.",
                ),
                PlanStep(
                    3,
                    "Verify code",
                    "Sandbox",
                    "Execute and validate the generated code locally.",
                ),
            ]

        elif task_type == TASK_FILE:

            steps = [
                PlanStep(
                    1,
                    "Collect required information",
                    "Local RAG",
                    "Retrieve supporting information from local documents.",
                ),
                PlanStep(
                    2,
                    "Generate content",
                    "Local Mistral 7B",
                    "Prepare the requested deliverable content.",
                ),
                PlanStep(
                    3,
                    "Create deliverable",
                    "Local File Tool",
                    "Generate the requested local file.",
                ),
                PlanStep(
                    4,
                    "Verify output",
                    "Local Validator",
                    "Check that the generated file exists and is readable.",
                ),
            ]

        elif task_type == TASK_CALCULATION:

            steps = [
                PlanStep(
                    1,
                    "Understand calculation",
                    "Local Mistral 7B",
                    "Determine the required calculation.",
                ),
                PlanStep(
                    2,
                    "Perform calculation",
                    "Local Calculator",
                    "Execute the calculation locally.",
                ),
                PlanStep(
                    3,
                    "Explain result",
                    "Local Mistral 7B",
                    "Present the calculation and result clearly.",
                ),
            ]

        else:

            steps = [
                PlanStep(
                    1,
                    "Understand request",
                    "Local Mistral 7B",
                    "Analyze the user's request locally.",
                ),
                PlanStep(
                    2,
                    "Generate response",
                    "Local Mistral 7B",
                    "Produce a local response.",
                ),
            ]

        return AgentPlan(
            task_type=task_type,
            objective=query,
            steps=steps,
        )


# ============================================================
# GLOBAL ROUTER
# ============================================================

router = TaskRouter()


# ============================================================
# PUBLIC API
# ============================================================

def classify_task(query: str) -> str:
    return router.classify(query)


def create_plan(query: str) -> AgentPlan:
    return router.create_plan(query)


def plan_to_text(plan: AgentPlan) -> str:

    lines = [
        f"Task Type: {plan.task_type}",
        f"Objective: {plan.objective}",
        "",
        "Execution Plan:",
    ]

    for step in plan.steps:

        lines.append(
            f"{step.step_number}. "
            f"{step.action} "
            f"[Tool: {step.tool}]"
        )

        lines.append(
            f"   {step.description}"
        )

    return "\n".join(lines)


# ============================================================
# DIAGNOSTIC
# ============================================================

def diagnostic():

    print("=" * 60)
    print("SOVEREIGN AGENT DIAGNOSTIC")
    print("=" * 60)

    tests = [
        "What is Python used for?",
        "Summarize the inspection report.",
        "According to the inspection report, what are the key findings?",
        "Analyze this scanned P&ID.",
        "Write Python code to calculate average marks.",
        "Create an approval note as a Word document.",
        "Calculate the percentage of completed work.",
        "What are the main points in the report?",
        "Who is the president of Brazil?",
    ]

    expected = [
        TASK_RAG,
        TASK_SUMMARY,
        TASK_RAG,
        TASK_IMAGE,
        TASK_CODE,
        TASK_FILE,
        TASK_CALCULATION,
        TASK_SUMMARY,
        TASK_RAG,
    ]

    passed = 0

    for query, expected_type in zip(tests, expected):

        plan = create_plan(query)

        status = "PASS" if plan.task_type == expected_type else "FAIL"

        if status == "PASS":
            passed += 1

        print()
        print("QUERY:", query)
        print("EXPECTED:", expected_type)
        print("TYPE:", plan.task_type)
        print("STATUS:", status)

        for step in plan.steps:
            print(
                f"  {step.step_number}. "
                f"{step.action} -> {step.tool}"
            )

    print()
    print("=" * 60)
    print(f"ROUTER TEST: {passed}/{len(tests)} PASSED")

    if passed == len(tests):
        print("Agent router: OK")
    else:
        print("Agent router: REVIEW REQUIRED")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    diagnostic()