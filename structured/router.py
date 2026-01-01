from structured.executor import query_structured_data
from rag.pipeline import query_blocks


def route_question(question: str, compliance_id: str, rag_store):
    # 1️⃣ Always try structured first
    structured_answer = query_structured_data(question, compliance_id)

    if structured_answer not in [
        "Not found in provided structured documents.",
        "No structured documents found."
    ]:
        return structured_answer

    # 2️⃣ Fall back to RAG
    if rag_store is not None:
        return query_blocks(rag_store, question, compliance_id)

    return "No relevant information found in uploaded documents."
