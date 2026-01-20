import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_intent(question: str,
    df_columns: list,
    column_samples: dict) -> dict:
    """
    Uses OpenAI to convert a natural language question
    into a structured query plan based on available columns.

    This is schema-aware and works for ANY Excel / CSV.

    You are also given sample values for each column.
    These samples represent real row values from the dataset.

    If the user mentions a value that appears in column samples,
    use that column as the filter_column.


    """

    system_prompt = """
You are an expert data analyst.

You are given:
- A list of column names from a spreadsheet
- A user question

Your task:
- Decide if the question can be answered using the columns
- Use SEMANTIC understanding, not exact string matching
- Column names may vary in casing, spacing, abbreviations, or wording
- If a column closely matches the intent, use it

Examples of semantic matching:
- "opening stock" → "Opening Stock", "Opening Balance", "Opening Qty"
- "closing stock" → "Closing Stock", "Balance"
- "quantity" → "Qty", "Quantity", "Stock"
- "item" → "Item Name", "Product", "Material"
- "date" → "Month", "Period", "Date"

Allowed operations:
- lookup        (single value for an entity)
- sum           (aggregate numeric column)
- count         (number of rows)
- average       (mean of numeric column)
- select_all    (return all values of a column)
- describe      (explain what the dataset represents)

Examples:
- "give me all product names" → select_all on Product Name
- "list all items" → select_all on Item Name
- "what does this data represent" → describe
- "what kind of data is this" → describe

Important:
- If the user mentions a year (e.g., 2005, 2007), and any column contains numeric values resembling years,
  treat that column as a filter_column.
- Years may appear as numbers or as column headers.



Return ONLY valid JSON.
DO NOT explain anything.
DO NOT add text outside JSON.

JSON format:
{
  "operation": "lookup | sum | count | average | not_applicable",
  "target_column": "<best matching column>",
  "filter_column": "<column used to filter rows>",
  "filter_value": "<value to filter on>"
}

Rules:
- If a reasonable mapping exists, DO NOT return not_applicable
- Return not_applicable ONLY if the columns clearly cannot answer the question
"""

    user_prompt = f"""
Available columns:
{df_columns}

Sample values per column:
{column_samples}

User question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"operation": "not_applicable"}