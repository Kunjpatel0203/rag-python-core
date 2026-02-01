# import re


# def normalize(text):
#     return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# def detect_numeric_operation(question):
#     q = question.lower()

#     if "sum" in q or "total" in q:
#         return "sum"
#     if "average" in q or "mean" in q:
#         return "average"
#     if "count" in q or "how many" in q:
#         return "count"

#     return "lookup"


# def pick_best_numeric_column(question, numeric_payloads):
#     q = normalize(question)

#     all_cols = set()
#     for payload in numeric_payloads:
#         all_cols.update(payload.keys())

#     best_col = None
#     best_score = 0

#     for col in all_cols:
#         col_norm = normalize(col)
#         score = sum(1 for w in q.split() if w in col_norm)

#         if score > best_score:
#             best_score = score
#             best_col = col

#     return best_col


# def filter_rows_by_entity(question, retrieved_metadata):
#     """
#     Keep only rows whose row_text contains an entity mentioned in question.
#     Example: 'desk lamp' must match row_text.
#     """

#     q = normalize(question)

#     filtered = []
#     for meta in retrieved_metadata:
#         row_text = normalize(meta.get("row_text", ""))
#         # if any multi-word product appears in question
#         # simple containment check
#         for token_len in range(1, 4):
#             # build sliding window of question tokens
#             words = q.split()
#             for i in range(len(words) - token_len + 1):
#                 phrase = " ".join(words[i:i+token_len])
#                 if phrase in row_text:
#                     filtered.append(meta)
#                     break

#     # If entity-specific rows found, use them
#     return filtered if filtered else retrieved_metadata


# def numeric_aggregate(retrieved_metadata, question):

#     # Step 1: filter rows if question mentions an entity
#     filtered_metadata = filter_rows_by_entity(question, retrieved_metadata)

#     numeric_payloads = [
#         meta["numeric"]
#         for meta in filtered_metadata
#         if "numeric" in meta and meta["numeric"]
#     ]

#     if not numeric_payloads:
#         return None

#     operation = detect_numeric_operation(question)
#     target_col = pick_best_numeric_column(question, numeric_payloads)

#     if not target_col:
#         return None

#     values = [
#         payload[target_col]
#         for payload in numeric_payloads
#         if target_col in payload
#     ]

#     if not values:
#         return None

#     # --- Execute ---
#     if operation == "sum":
#         return f"{target_col} total is {sum(values)}"

#     if operation == "average":
#         return f"{target_col} average is {sum(values)/len(values)}"

#     if operation == "count":
#         return f"Count is {len(values)}"

#     # lookup
#     return f"{target_col} is {values[0]}"













# import re


# def normalize(text):
#     return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# # ----------------------------
# # Detect operation type
# # ----------------------------
# def detect_operation(question):
#     q = question.lower()

#     if any(k in q for k in ["sum", "total", "overall"]):
#         return "sum"

#     if any(k in q for k in ["average", "mean"]):
#         return "average"

#     if any(k in q for k in ["count", "how many"]):
#         return "count"

#     # lookup numeric single-value
#     return "lookup"


# # ----------------------------
# # Decide if question is numeric
# # ----------------------------
# def question_requests_numeric(question):
#     q = question.lower()
#     numeric_words = [

#     # --- Quantity / Counting ---
#     "count", "number", "qty", "quantity", "units", "items", "entries",
#     "records", "rows", "instances", "occurrences", "frequency",

#     # --- Totals / Aggregates ---
#     "total", "sum", "overall", "grand", "aggregate", "combined", "cumulative",

#     # --- Price / Cost ---
#     "price", "cost", "amount", "value", "rate", "charge", "fee", "fare",
#     "tariff", "expense", "expenditure", "spend", "spending",

#     # --- Financial ---
#     "revenue", "income", "profit", "loss", "margin", "turnover",
#     "sales", "earning", "balance", "budget", "capital",
#     "investment", "fund", "funds", "cash", "flow", "credit", "debit",

#     # --- Stock / Inventory ---
#     "stock", "inventory", "opening", "closing", "balance",
#     "available", "remaining", "hand", "inhand", "onhand",
#     "sold", "purchased", "supply", "demand",

#     # --- Measurements ---
#     "weight", "height", "length", "width", "size", "volume",
#     "area", "distance", "duration", "time", "speed", "capacity",

#     # --- Statistics ---
#     "average", "mean", "median", "mode", "min", "minimum",
#     "max", "maximum", "range", "variance", "deviation", "std",

#     # --- Performance / Metrics ---
#     "score", "rating", "rank", "grade", "percentage", "percent", "%",
#     "ratio", "index", "level", "threshold", "target", "actual",

#     # --- Dates / Time counts ---
#     "days", "months", "years", "period", "quarter", "week",

#     # --- HR / People ---
#     "employees", "staff", "workers", "headcount", "attendance",
#     "absent", "present", "hours", "overtime", "salary", "wage", "pay",

#     # --- Production / Ops ---
#     "output", "input", "yield", "defect", "error", "downtime",
#     "efficiency", "utilization", "capacity",

#     # --- Science / Research ---
#     "value", "reading", "measurement", "observation", "sample",
#     "trial", "result",

#     # --- Generic math triggers ---
#     "calculate", "compute", "sum", "add", "subtract",
#     "multiply", "divide", "difference",

#     # --- Common synonyms ---
#     "how many", "how much", "total of", "sum of", "amount of"
#     ]

#     return any(w in q for w in numeric_words)


# # ----------------------------
# # Exact entity filter
# # ----------------------------
# def filter_rows_by_entity(question, retrieved_metadata):
#     q = normalize(question)

#     # if question contains "all products", do not filter
#     if "all product" in q or "all items" in q:
#         return retrieved_metadata

#     filtered = []
#     for meta in retrieved_metadata:
#         row_text = normalize(meta.get("row_text", ""))

#         # extract product name from row_text
#         # row_text always starts with: "product name: xxx."
#         if "product name:" in row_text:
#             product = row_text.split("product name:")[1].split(".")[0].strip()
#             if product in q:
#                 filtered.append(meta)

#     return filtered if filtered else retrieved_metadata


# # ----------------------------
# # Pick correct numeric column
# # ----------------------------
# def pick_best_numeric_column(question, numeric_payloads):
#     q = normalize(question)

#     all_cols = set()
#     for payload in numeric_payloads:
#         all_cols.update(payload.keys())

#     best_col = None
#     best_score = 0

#     for col in all_cols:
#         col_norm = normalize(col)
#         score = sum(1 for w in q.split() if w in col_norm)
#         if score > best_score:
#             best_score = score
#             best_col = col

#     return best_col


# # ----------------------------
# # Main numeric aggregator
# # ----------------------------
# def numeric_aggregate(retrieved_metadata, question):

#     # If question is not numeric in nature → let normal RAG handle
#     if not question_requests_numeric(question):
#         return None

#     # Filter rows by entity
#     filtered_metadata = filter_rows_by_entity(question, retrieved_metadata)

#     numeric_payloads = [
#         meta["numeric"]
#         for meta in filtered_metadata
#         if "numeric" in meta and meta["numeric"]
#     ]

#     if not numeric_payloads:
#         return None

#     operation = detect_operation(question)
#     target_col = pick_best_numeric_column(question, numeric_payloads)

#     if not target_col:
#         return None

#     values = [
#         payload[target_col]
#         for payload in numeric_payloads
#         if target_col in payload
#     ]

#     if not values:
#         return None

#     # ---------------- Execute ----------------
#     if operation == "sum":
#         return f"{target_col} total is {sum(values)}"

#     if operation == "average":
#         return f"{target_col} average is {sum(values)/len(values)}"

#     if operation == "count":
#         return f"Count is {len(values)}"

#     # lookup (single row)
#     return f"{target_col} is {values[0]}"












import re
import pandas as pd


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# ----------------------------
# Detect numeric operation
# ----------------------------
def detect_operation(question):
    q = question.lower()
    if any(k in q for k in ["sum", "total", "overall"]):
        return "sum"
    if any(k in q for k in ["average", "mean"]):
        return "average"
    if any(k in q for k in ["count", "how many"]):
        return "count"
    return "lookup"


# ----------------------------
# Decide if question is numeric
# ----------------------------
def question_requests_numeric(question):
    q = question.lower()
    numeric_words = [
        "total", "sum", "average", "mean", "count", "how many",
        "price", "cost", "amount", "value",
        "stock", "sold", "units", "quantity", "number",
        "revenue", "profit", "loss", "income", "expense",
        "balance", "opening", "closing"
    ]
    return any(w in q for w in numeric_words)


# ----------------------------
# Detect entity from FAISS rows
# ----------------------------
def detect_entity_from_rows(question, retrieved_metadata):
    q = normalize(question)

    for meta in retrieved_metadata:
        row_text = normalize(meta.get("row_text", ""))

        if "product name:" in row_text:
            product = row_text.split("product name:")[1].split(".")[0].strip()
            if product in q:
                return product

    return None


# ----------------------------
# Pick numeric column from dataframe
# ----------------------------
def pick_best_numeric_column(question, df_columns):
    q = normalize(question)

    # 1. Direct phrase containment (deterministic)
    for col in df_columns:
        col_norm = normalize(col)
        if col_norm in q:
            return col

    # 2. Token overlap fallback
    best_col = None
    best_score = 0
    for col in df_columns:
        col_norm = normalize(col)
        score = sum(1 for w in q.split() if w in col_norm)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col



# ----------------------------
# MAIN NUMERIC AGGREGATOR
# ----------------------------
def numeric_aggregate(retrieved_metadata, question, structured_excel_path):

    # If not numeric-type question → let normal RAG handle
    if not question_requests_numeric(question):
        return None

    # Load FULL dataframe for correct math
    df = pd.read_excel(structured_excel_path)
    df.columns = [normalize(c) for c in df.columns]

    operation = detect_operation(question)

    # Pick correct numeric column
    target_col = pick_best_numeric_column(question, df.columns)
    if not target_col or target_col not in df.columns:
        return None

    # Detect entity from FAISS retrieved rows
    entity = detect_entity_from_rows(question, retrieved_metadata)

    # ---------------- ENTITY LOOKUP ----------------
    if entity:
        # find product/name column
        name_col = None
        for col in df.columns:
            if "name" in col or "product" in col or "item" in col:
                name_col = col
                break

        if not name_col:
            return None

        row = df[df[name_col].astype(str).str.lower() == entity.lower()]
        if row.empty:
            return None

        value = row.iloc[0][target_col]
        return f"{target_col} is {value}"

    # ---------------- GLOBAL AGGREGATION ----------------
    if operation == "sum":
        total = pd.to_numeric(df[target_col], errors="coerce").sum()
        return f"{target_col} total is {total}"

    if operation == "average":
        avg = pd.to_numeric(df[target_col], errors="coerce").mean()
        return f"{target_col} average is {avg}"

    if operation == "count":
        return f"Count is {len(df)}"

    return None
