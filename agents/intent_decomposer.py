import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def decompose_intent(question: str) -> dict:
    """
    Decomposes a complex user question into independent sub-questions.
    Each sub-question is classified as structured or unstructured.
    """

    system_prompt = """
You are an expert AI planner.

Your task:
- Break a user question into independent sub-questions
- Each sub-question must be answerable on its own
- Classify each sub-question as:
  - "structured" (Excel / CSV / SQL type data)
  - "unstructured" (PDF / DOC / URL text data)

Rules:
- If the question has only one intent, return one sub-question
- Do NOT answer the questions
- Do NOT merge questions
- Do NOT invent new information
- Return ONLY valid JSON

Output format:
{
  "sub_questions": [
    {
      "type": "structured | unstructured",
      "question": "..."
    }
  ]
}
"""

    user_prompt = f"""
User question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except Exception:
        # Safe fallback: treat entire question as unstructured
        return {
            "sub_questions": [
                {"type": "unstructured", "question": question}
            ]
        }
