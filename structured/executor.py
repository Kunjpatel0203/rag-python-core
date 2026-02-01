# import os
# import pandas as pd
# import re
# from structured.nlq import extract_intent




# # -----------------------------
# # Utility: column sample values
# # -----------------------------
# def get_column_samples(df, max_samples=5):
#     samples = {}
#     for col in df.columns:
#         values = (
#             df[col]
#             .dropna()
#             .astype(str)
#             .unique()
#             .tolist()
#         )
#         samples[col] = values[:max_samples]
#     return samples


# # -----------------------------
# # Detect header row
# # -----------------------------
# def detect_header_row(df: pd.DataFrame, max_rows: int = 10) -> int:
#     best_row = 0
#     best_score = 0

#     for i in range(min(max_rows, len(df))):
#         row = df.iloc[i]
#         score = sum(
#             isinstance(cell, str) and cell.strip() != ""
#             for cell in row
#         )
#         if score > best_score:
#             best_score = score
#             best_row = i

#     return best_row


# # -----------------------------
# # Load all sheets (Excel)
# # -----------------------------
# def load_all_sheets_with_detected_header(path: str):
#     sheets = pd.read_excel(path, sheet_name=None, header=None)
#     dataframes = []

#     for sheet_name, raw_df in sheets.items():
#         raw_df = raw_df.dropna(how="all")
#         if raw_df.empty:
#             continue

#         header_row = detect_header_row(raw_df)

#         df = raw_df.iloc[header_row + 1:].copy()
#         df.columns = raw_df.iloc[header_row].astype(str).str.strip()

#         df = df.loc[:, df.columns != ""]
#         df = df.dropna(how="all")

#         if not df.empty:
#             dataframes.append((sheet_name, df))

#     return dataframes


# # -----------------------------
# # MAIN STRUCTURED QUERY ROUTER
# # -----------------------------
# def query_structured_data(question: str, compliance_id: str):
#     base = f"data/structured/compliance_{compliance_id}"

#     if not os.path.exists(base):
#         return "No structured documents found."

#     for file in os.listdir(base):
#         path = os.path.join(base, file)

#         try:
#             # ---------- Excel ----------
#             if file.endswith(".xlsx"):
#                 sheets = load_all_sheets_with_detected_header(path)

#                 for sheet_name, df in sheets:
#                     print(f"\n[DETECTED SHEET] {sheet_name}")
#                     print("[DETECTED COLUMNS]", df.columns.tolist())

#                     column_samples = get_column_samples(df)

#                     plan = extract_intent(
#                         question,
#                         df.columns.tolist(),
#                         column_samples
#                     )

#                     print("[NLQ PLAN]", plan)

#                     if plan.get("operation") == "not_applicable":
#                         continue

#                     result = execute_plan(df, plan)
#                     return f"[Sheet: {sheet_name}]\n{result}"

#             # ---------- CSV ----------
#             elif file.endswith(".csv"):
#                 df = pd.read_csv(path)

#                 print("[DETECTED COLUMNS]", df.columns.tolist())

#                 column_samples = get_column_samples(df)

#                 plan = extract_intent(
#                     question,
#                     df.columns.tolist(),
#                     column_samples
#                 )

#                 print("[NLQ PLAN]", plan)

#                 if plan.get("operation") == "not_applicable":
#                     continue

#                 return execute_plan(df, plan)

#         except Exception as e:
#             print("[FILE LOAD ERROR]", e)
#             continue

#     return "No relevant information found in uploaded documents."


# # -----------------------------
# # EXECUTION ENGINE
# # -----------------------------
# def execute_plan(df: pd.DataFrame, plan: dict):
#     operation = plan.get("operation")

#     def normalize(text):
#         return re.sub(r"[^a-z0-9]+", "", str(text).lower())

#     def month_aware_match(cell_value, filter_value):
#         c = normalize(cell_value)
#         f = normalize(filter_value)

#         # direct or partial match
#         if f in c or c in f:
#             return True

#         # universal month mapping (GENERALIZED)
#         month_map = {
#             "jan": "january",
#             "feb": "february",
#             "mar": "march",
#             "apr": "april",
#             "may": "may",
#             "jun": "june",
#             "jul": "july",
#             "aug": "august",
#             "sep": "september",
#             "oct": "october",
#             "nov": "november",
#             "dec": "december",
#         }

#         for short, full in month_map.items():
#             if c == short and f == full:
#                 return True
#             if c == full and f == short:
#                 return True

#         return False

#     # -------- lookup --------
#     if operation == "lookup":
#         fcol = plan.get("filter_column")
#         fval = plan.get("filter_value")
#         tcol = plan.get("target_column")

#         if not fcol or not fval or not tcol:
#             return "Invalid lookup parameters."

#         match = df[
#             df[fcol]
#             .astype(str)
#             .apply(lambda x: month_aware_match(x, fval))
#         ]

#         if match.empty:
#             return "No matching record found."

#         return f"{tcol} is {match.iloc[0][tcol]}."

#     # -------- sum --------
#     if operation == "sum":
#         total = pd.to_numeric(df[plan["target_column"]], errors="coerce").sum()
#         return f"Sum of {plan['target_column']} is {total}."

#     # -------- count --------
#     if operation == "count":
#         return f"Count is {len(df)}."

#     # -------- average --------
#     if operation == "average":
#         avg = pd.to_numeric(df[plan["target_column"]], errors="coerce").mean()
#         return f"Average of {plan['target_column']} is {avg}."

#     # -------- select_all --------
#     if operation == "select_all":
#         values = (
#             df[plan["target_column"]]
#             .dropna()
#             .astype(str)
#             .unique()
#             .tolist()
#         )
#         return f"{plan['target_column']} values: {', '.join(values)}"

#     # -------- describe --------
#     if operation == "describe":
#         return (
#             f"This dataset contains {len(df)} records. "
#             f"Columns: {', '.join(df.columns.tolist())}."
#         )

#     return "Operation not supported."














# import os
# import pandas as pd
# import re
# from structured.nlq import extract_intent, load_dataframe, get_column_samples, normalize


# # -----------------------------
# # Detect header row for messy Excel
# # -----------------------------
# def detect_header_row(df: pd.DataFrame, max_rows: int = 10) -> int:
#     best_row = 0
#     best_score = 0

#     for i in range(min(max_rows, len(df))):
#         row = df.iloc[i]
#         score = sum(
#             isinstance(cell, str) and cell.strip() != ""
#             for cell in row
#         )
#         if score > best_score:
#             best_score = score
#             best_row = i

#     return best_row


# # -----------------------------
# # Load all sheets with header detection
# # -----------------------------
# def load_all_sheets_with_detected_header(path: str):
#     sheets = pd.read_excel(path, sheet_name=None, header=None)
#     dataframes = []

#     for sheet_name, raw_df in sheets.items():
#         raw_df = raw_df.dropna(how="all")
#         if raw_df.empty:
#             continue

#         header_row = detect_header_row(raw_df)

#         df = raw_df.iloc[header_row + 1:].copy()
#         df.columns = raw_df.iloc[header_row].astype(str)

#         df = df.loc[:, df.columns != ""]
#         df = df.dropna(how="all")

#         # 🔴 Normalize columns HERE
#         df.columns = [normalize(c) for c in df.columns]

#         if not df.empty:
#             dataframes.append((sheet_name, df))

#     return dataframes


# # -----------------------------
# # MAIN STRUCTURED QUERY ROUTER
# # -----------------------------
# def query_structured_data(question: str, compliance_id: str):
#     base = f"data/structured/compliance_{compliance_id}"

#     if not os.path.exists(base):
#         return "No structured documents found."

#     for file in os.listdir(base):
#         path = os.path.join(base, file)

#         try:
#             # ---------- Excel ----------
#             if file.endswith(".xlsx"):
#                 sheets = load_all_sheets_with_detected_header(path)

#                 for sheet_name, df in sheets:
#                     print(f"\n[DETECTED SHEET] {sheet_name}")
#                     print("[DETECTED COLUMNS]", df.columns.tolist())

#                     column_samples = get_column_samples(df)

#                     # ---------- Hard rule: if a sample value appears in question, force lookup ----------
#                     for col, samples in column_samples.items():
#                         for s in samples:
#                             if s.lower() in question.lower():
#                                 plan = {
#                                     "operation": "lookup",
#                                     "target_column": "cost price total (usd)",
#                                     "filter_column": col,
#                                     "filter_value": s
#                                 }
#                                 break
#                         else:
#                             continue
#                         break
#                     else:
#                         plan = extract_intent(question, df.columns.tolist(), column_samples)


#                     print("[NLQ PLAN]", plan)

#                     if plan.get("operation") == "not_applicable":
#                         continue

#                     result = execute_plan(df, plan)
#                     return f"[Sheet: {sheet_name}]\n{result}"

#             # ---------- CSV ----------
#             elif file.endswith(".csv"):
#                 df = pd.read_csv(path)
#                 df.columns = [normalize(c) for c in df.columns]

#                 print("[DETECTED COLUMNS]", df.columns.tolist())

#                 column_samples = get_column_samples(df)

#                 plan = extract_intent(
#                     question,
#                     df.columns.tolist(),
#                     column_samples
#                 )

#                 print("[NLQ PLAN]", plan)

#                 if plan.get("operation") == "not_applicable":
#                     continue

#                 return execute_plan(df, plan)

#         except Exception as e:
#             print("[FILE LOAD ERROR]", e)
#             continue

#     return "No relevant information found in uploaded documents."


# # -----------------------------
# # EXECUTION ENGINE
# # -----------------------------
# def execute_plan(df: pd.DataFrame, plan: dict):
#     operation = plan.get("operation")

#     target_col = normalize(plan.get("target_column"))
#     filter_col = normalize(plan.get("filter_column"))
#     filter_value = plan.get("filter_value")

#     # -------- lookup --------
#     if operation == "lookup":
#         if target_col not in df.columns or filter_col not in df.columns:
#             return "Column mapping failed."

#         mask = df[filter_col].astype(str).str.lower() == str(filter_value).lower()
#         result = df.loc[mask, target_col]

#         if result.empty:
#             return "No matching record found."

#         return f"{plan['target_column']} is {result.iloc[0]}."

#     # -------- sum --------
#     if operation == "sum":
#         total = pd.to_numeric(df[target_col], errors="coerce").sum()
#         return f"Sum of {plan['target_column']} is {total}."

#     # -------- count --------
#     if operation == "count":
#         return f"Count is {len(df)}."

#     # -------- average --------
#     if operation == "average":
#         avg = pd.to_numeric(df[target_col], errors="coerce").mean()
#         return f"Average of {plan['target_column']} is {avg}."

#     # -------- select_all --------
#     if operation == "select_all":
#         values = df[target_col].dropna().astype(str).unique().tolist()
#         return f"{plan['target_column']} values: {', '.join(values)}"

#     # -------- describe --------
#     if operation == "describe":
#         return (
#             f"This dataset contains {len(df)} records. "
#             f"Columns: {', '.join(df.columns.tolist())}."
#         )

#     return "Operation not supported."













import os
import pandas as pd
from openai import OpenAI

from structured.nlq import get_column_samples, normalize
from structured.executor_header_detect import load_all_sheets_with_detected_header

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# Detect filter column + value using sample values
# =====================================================
def detect_filter(question, column_samples):
    q = normalize(question)

    for col, samples in column_samples.items():
        for s in samples:
            s_norm = normalize(s)

            # match full words only
            if not s_norm:
                continue

            # exact phrase match inside question
            if s_norm in q:
                return col, s

            # also allow partial token overlap for multi-word values
            s_tokens = set(s_norm.split())
            q_tokens = set(q.split())

            if s_tokens.issubset(q_tokens):
                return col, s

    return None, None



# =====================================================
# Detect target column using embeddings
# =====================================================
def detect_target_column(question, columns):
    col_emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=columns
    )
    col_embeddings = [e.embedding for e in col_emb_response.data]

    q_emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    q_embedding = q_emb_response.data[0].embedding

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    best_idx = 0
    best_score = -1

    for i, emb in enumerate(col_embeddings):
        score = cosine(q_embedding, emb)
        if score > best_score:
            best_score = score
            best_idx = i

    return columns[best_idx]


# =====================================================
# Detect operation from question
# =====================================================
def detect_operation(question):
    q = question.lower()

    if "sum" in q or "total of all" in q:
        return "sum"
    if "average" in q or "mean" in q:
        return "average"
    if "list" in q or "show all" in q:
        return "select_all"

    return "lookup"


# =====================================================
# Execute query on dataframe
# =====================================================
def execute_query(df, question):
    df.columns = [normalize(c) for c in df.columns]

    columns = df.columns.tolist()
    column_samples = get_column_samples(df)

    print("[DETECTED COLUMNS]", columns)

    # Step 1: detect filter
    filter_col, filter_val = detect_filter(question, column_samples)

    # Step 2: detect target
    target_col = detect_target_column(question, columns)

    # Step 3: detect operation
    operation = detect_operation(question)

    # -------------------------
    # Execute operations
    # -------------------------

    # select_all
    if operation == "select_all":
        values = df[target_col].dropna().astype(str).unique().tolist()
        return f"{target_col} values: {', '.join(values)}"

    # sum
    if operation == "sum":
        total = pd.to_numeric(df[target_col], errors="coerce").sum()
        return f"Sum of {target_col} is {total}"

    # average
    if operation == "average":
        avg = pd.to_numeric(df[target_col], errors="coerce").mean()
        return f"Average of {target_col} is {avg}"

    # lookup
    if filter_col is None:
        return "Could not detect filter value in question."

    mask = df[filter_col].astype(str).str.lower() == str(filter_val).lower()
    result = df.loc[mask, target_col]

    if result.empty:
        return "No matching record found."

    return f"{target_col} is {result.iloc[0]}"


# =====================================================
# MAIN ROUTER
# =====================================================
def query_structured_data(question: str, compliance_id: str):
    base = f"data/structured/compliance_{compliance_id}"

    if not os.path.exists(base):
        return "No structured documents found."

    for file in os.listdir(base):
        path = os.path.join(base, file)

        try:
            # Excel with automatic header detection
            if file.endswith(".xlsx"):
                sheets = load_all_sheets_with_detected_header(path)

                for sheet_name, df in sheets:
                    print(f"\n[DETECTED SHEET] {sheet_name}")
                    result = execute_query(df, question)
                    return f"[Sheet: {sheet_name}]\n{result}"

            # CSV
            if file.endswith(".csv"):
                df = pd.read_csv(path)
                result = execute_query(df, question)
                return result

        except Exception as e:
            print("[STRUCTURED ERROR]", e)
            continue

    return "No relevant structured data found."
