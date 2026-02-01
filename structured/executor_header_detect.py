import pandas as pd


def detect_header_row(df, max_rows=10):
    best_row = 0
    best_score = 0

    for i in range(min(max_rows, len(df))):
        row = df.iloc[i]
        score = sum(
            isinstance(cell, str) and cell.strip() != ""
            for cell in row
        )
        if score > best_score:
            best_score = score
            best_row = i

    return best_row


def load_all_sheets_with_detected_header(path):
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    results = []

    for sheet_name, raw_df in sheets.items():
        raw_df = raw_df.dropna(how="all")
        if raw_df.empty:
            continue

        header_row = detect_header_row(raw_df)

        df = raw_df.iloc[header_row + 1:].copy()
        df.columns = raw_df.iloc[header_row].astype(str)

        df = df.loc[:, df.columns != ""]
        df = df.dropna(how="all")

        if not df.empty:
            results.append((sheet_name, df))

    return results
