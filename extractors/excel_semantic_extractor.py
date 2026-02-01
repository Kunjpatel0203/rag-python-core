import pandas as pd
from structured.executor_header_detect import load_all_sheets_with_detected_header
from structured.nlq import normalize


def excel_semantic_extractor(file_path):
    """
    Returns:
    row_text_chunks: list[str]
    row_metadata: list[dict] containing original row data
    """

    row_text_chunks = []
    row_metadata = []

    sheets = load_all_sheets_with_detected_header(file_path)

    for sheet_name, df in sheets:
        # Normalize column headers
        df.columns = [normalize(c) for c in df.columns]

        # Clean all string cells
        df = df.applymap(lambda x: normalize(x) if isinstance(x, str) else x)

        for idx, row in df.iterrows():
            text_parts = []
            numeric_payload = {}

            for col in df.columns:
                val = row[col]

                if pd.isna(val):
                    continue

                # Build semantic text
                text_parts.append(f"{col}: {val}")

                # Store numeric fields separately
                if isinstance(val, (int, float)):
                    numeric_payload[col] = val

            if not text_parts:
                continue

            chunk_text = ". ".join(text_parts) + "."

            row_text_chunks.append(chunk_text)

            row_metadata.append({
                "sheet": sheet_name,
                "row_index": int(idx),
                "numeric": numeric_payload,
                "row_text": chunk_text
            })

    return row_text_chunks, row_metadata
