import os
import pandas as pd
from unstructured.partition.auto import partition


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    # ==========================
    # ✅ GENERIC EXCEL HANDLER
    # ==========================
    if ext in [".xlsx", ".xls"]:
        return excel_to_facts(file_path)

    # ==========================
    # DEFAULT: PDF / DOC / PPT
    # ==========================
    elements = partition(filename=file_path)
    return "\n".join(el.text for el in elements if el.text)


def excel_to_facts(file_path: str) -> str:
    """
    Converts ANY Excel table into natural-language facts.
    Works for:
    - unknown schemas
    - merged cells
    - multi-row headers
    """

    df = pd.read_excel(file_path, header=None)

    # 1️⃣ Handle merged cells
    df = df.ffill().bfill()

    # 2️⃣ Detect header row
    header_row = None
    for i in range(len(df)):
        text_cells = sum(isinstance(x, str) for x in df.iloc[i])
        if text_cells >= len(df.columns) // 2:
            header_row = i
            break

    if header_row is None:
        header_row = 0

    headers = df.iloc[header_row].astype(str).tolist()
    data = df.iloc[header_row + 1:]
    data.columns = headers

    facts = []

    # 3️⃣ Convert each cell into a fact
    for _, row in data.iterrows():
        row_context = []

        # Try to capture row context (Month / Date / Period)
        for col in headers:
            if col.upper() in ["MONTH", "DATE", "PERIOD", "YEAR"] and pd.notna(row[col]):
                row_context.append(f"{col} {row[col]}")

        context = ", ".join(row_context) if row_context else "This record"

        for col, val in row.items():
            if pd.notna(val):
                facts.append(f"{context}: {col} is {val}.")

    return "\n".join(facts)
