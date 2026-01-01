import pandas as pd


def extract_excel_metadata(file_path: str) -> str:
    """
    Extracts non-tabular / merged-cell content from ALL sheets of an Excel file.
    Returns clean text for RAG ingestion.
    """

    sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    collected_text = []

    for sheet_name, df in sheets.items():
        df = df.dropna(how="all")

        if df.empty:
            continue

        collected_text.append(f"Sheet: {sheet_name}")

        for _, row in df.iterrows():
            row_text = " ".join(
                str(cell).strip()
                for cell in row
                if isinstance(cell, str) and cell.strip()
            )

            # Ignore pure numeric rows (table data)
            if row_text and not row_text.replace(".", "").replace(" ", "").isdigit():
                collected_text.append(row_text)

    return "\n".join(collected_text)
