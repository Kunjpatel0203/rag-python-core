from agents.intent_decomposer import decompose_intent
from structured.executor import query_structured_data
from rag.pipeline import query_blocks


def detect_question_type(question: str) -> str:
    """
    Detects the intent type of a question.
    This is deterministic and fast (no LLM).
    """

    q = question.lower()

    list_keywords = [
        "give all", "list", "names of", "categories", "types of",
        "all the", "enumerate"
    ]

    how_keywords = [
        "how", "how was", "how does", "how did"
    ]

    lookup_keywords = [
        "what is", "what are", "value of", "amount of"
    ]

    for kw in list_keywords:
        if kw in q:
            return "LIST"

    for kw in how_keywords:
        if q.startswith(kw):
            return "HOW"

    for kw in lookup_keywords:
        if q.startswith(kw):
            return "LOOKUP"

    return "EXPLAIN"


def run_phase2(question, compliance_id, rag_store):
    """
    STRUCTURED-FIRST hybrid execution with QUESTION-TYPE AWARENESS.
    """

    plan = decompose_intent(question)
    answers = []

    for idx, item in enumerate(plan["sub_questions"], start=1):
        sub_q = item["question"]

        q_type = detect_question_type(sub_q)

        # 1️⃣ ALWAYS TRY STRUCTURED FIRST
        structured_ans = query_structured_data(sub_q, compliance_id)

        if structured_ans and not structured_ans.lower().startswith(
            ("no relevant", "not found", "no structured")
        ):
            answers.append(f"[Structured Answer {idx}]\n{structured_ans}")
            continue

        # 2️⃣ FALL BACK TO UNSTRUCTURED (RAG)
        if rag_store is None:
            answers.append(
                f"[Unstructured Answer {idx}]\nNo unstructured documents available."
            )
            continue

        # 🔴 SPECIAL HANDLING FOR LIST QUESTIONS
        if q_type == "LIST":
            rag_ans = query_blocks(
                rag_store,
                sub_q + " (provide a complete and exhaustive list)",
                compliance_id
            )
        else:
            rag_ans = query_blocks(rag_store, sub_q, compliance_id)

        if rag_ans and not rag_ans.lower().startswith(
            ("not found", "no unstructured")
        ):
            answers.append(f"[Unstructured Answer {idx}]\n{rag_ans}")
        else:
            answers.append(
                f"[Answer {idx}]\nNo relevant information found in uploaded documents."
            )

    return "\n\n".join(answers)
