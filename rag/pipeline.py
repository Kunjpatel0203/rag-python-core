# import os

# # ===== Existing loaders (DO NOT REMOVE) =====
# from loaders.url_loader import load_url
# from loaders.file_loader import download_file

# # ===== New browser-based loader (ADDED) =====
# from loaders.url_loader_browser import load_url_with_browser

# from extractors.universal_extractor import extract_text
# from extractors.excel_metadata_extractor import extract_excel_metadata

# from processing.chunker import chunk_text
# from processing.embedder import get_embedder
# from processing.llm import get_llm

# from vectorstore.faiss_store import create_vector_store, load_vector_store
# from rag.prompt import build_prompt
# from structured.store import save_structured_file


# # =============================
# # 🔧 CONFIG
# # =============================
# USE_BROWSER = True   # ✅ Toggle safely (True = Playwright, False = old loader)

# BASE_CHUNK_DIR = "data/chunks"
# BASE_EMBED_DIR = "data/embeddings"


# # =============================
# # 🧩 Helper: Save Chunks
# # =============================
# def save_chunks(compliance_id, block_id, source_name, chunks):
#     os.makedirs(f"{BASE_CHUNK_DIR}/compliance_{compliance_id}", exist_ok=True)

#     file_path = (
#         f"{BASE_CHUNK_DIR}/compliance_{compliance_id}/"
#         f"block_{block_id}_{source_name}.txt"
#     )

#     with open(file_path, "w", encoding="utf-8") as f:
#         f.write(f"COMPLIANCE ID: {compliance_id}\n")
#         f.write(f"BLOCK ID: {block_id}\n")
#         f.write(f"SOURCE: {source_name}\n")
#         f.write("=" * 50 + "\n\n")

#         for i, chunk in enumerate(chunks, 1):
#             f.write(f"[CHUNK {i}]\n{chunk}\n\n")


# # =============================
# # 🧠 INGEST BLOCKS
# # =============================
# def ingest_blocks(blocks, compliance_id):
#     embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"

#     # ✅ Reuse embeddings if present
#     if os.path.exists(f"{embed_path}/index.faiss"):
#         print("✔ Loading existing embeddings (no OpenAI cost)")
#         return load_vector_store(embed_path)

#     print("⏳ Creating embeddings...")

#     all_chunks = []
#     all_metadata = []

#     for block in blocks:
#         if not block.get("type"):
#             continue

#         block_type = block["type"].upper()
#         block_id = block["block_id"]
#         value = block["value"]

#         # ==================================================
#         # 📊 STRUCTURED FILES (EXCEL / CSV / GSHEET)
#         # ==================================================
#         if block_type in ["XLS", "XLSX", "CSV", "GSHEET"]:
#             # 1️⃣ Save for NLQ (structured)
#             save_structured_file(block, compliance_id)

#             # 2️⃣ Extract metadata / merged headers → RAG
#             try:
#                 metadata_text = extract_excel_metadata(value)
#             except Exception as e:
#                 print("[EXCEL METADATA ERROR]", e)
#                 continue

#             if metadata_text and metadata_text.strip():
#                 meta_chunks = chunk_text(metadata_text)
#                 source_name = (
#                     os.path.basename(value).replace(".", "_") + "_metadata"
#                 )

#                 save_chunks(compliance_id, block_id, source_name, meta_chunks)

#                 for c in meta_chunks:
#                     all_chunks.append(c)
#                     all_metadata.append({
#                         "compliance_id": compliance_id,
#                         "block_id": block_id,
#                         "source": value,
#                         "type": "excel_metadata"
#                     })

#             # 🚫 DO NOT fall through to unstructured handling
#             continue

#         # ==================================================
#         # 🌐 URL INGESTION (SAFE HYBRID)
#         # ==================================================
#         if block_type == "URL":
#             if USE_BROWSER:
#                 page_text, downloaded_docs = load_url_with_browser(value)
#             else:
#                 page_text, downloaded_docs = load_url(value)

#             # Page text → chunks
#             page_chunks = chunk_text(page_text)
#             save_chunks(compliance_id, block_id, "url_page", page_chunks)

#             for c in page_chunks:
#                 all_chunks.append(c)
#                 all_metadata.append({
#                     "compliance_id": compliance_id,
#                     "block_id": block_id,
#                     "source": value,
#                     "type": "url_page"
#                 })

#             # Downloaded docs (PDF / XLS / PPT etc.)
#             for path in downloaded_docs:
#                 try:
#                     text = extract_text(path)
#                 except Exception as e:
#                     print("[DOC EXTRACT ERROR]", e)
#                     continue

#                 doc_chunks = chunk_text(text)
#                 name = os.path.basename(path).replace(".", "_")

#                 save_chunks(compliance_id, block_id, name, doc_chunks)

#                 for c in doc_chunks:
#                     all_chunks.append(c)
#                     all_metadata.append({
#                         "compliance_id": compliance_id,
#                         "block_id": block_id,
#                         "source": path,
#                         "type": "url_document"
#                     })

#             continue

#         # ==================================================
#         # 📄 PDF / DOC / PPT (UNSTRUCTURED)
#         # ==================================================
#         try:
#             text = extract_text(value)
#         except Exception as e:
#             print("[FILE EXTRACT ERROR]", e)
#             continue

#         chunks = chunk_text(text)
#         name = os.path.basename(value).replace(".", "_")

#         save_chunks(compliance_id, block_id, name, chunks)

#         for c in chunks:
#             all_chunks.append(c)
#             all_metadata.append({
#                 "compliance_id": compliance_id,
#                 "block_id": block_id,
#                 "source": value,
#                 "type": "unstructured"
#             })

#     # ==================================================
#     # 🚫 No RAG content
#     # ==================================================
#     if not all_chunks:
#         print("ℹ No unstructured documents found. Skipping vector store creation.")
#         return None

#     # ==================================================
#     # 🧠 Create Vector Store
#     # ==================================================
#     embedder = get_embedder()
#     store = create_vector_store(all_chunks, all_metadata, embedder)

#     os.makedirs(embed_path, exist_ok=True)
#     store.save_local(embed_path)

#     print("✅ Embeddings saved locally")
#     return store


# # =============================
# # 🔍 QUERY UNSTRUCTURED (RAG)
# # =============================
# #1. 
# # def query_blocks(store, question, compliance_id):
# #     if store is None:
# #         return "No unstructured documents available for this compliance."

# #     # 1️⃣ Primary retrieval
# #     docs = store.similarity_search(
# #         question,
# #         k=10,
# #         filter={"compliance_id": compliance_id}
# #     )

# #     # 2️⃣ If nothing found, try semantic fallback
# #     if not docs:
# #         semantic_fallback = f"{question} related information"
# #         docs = store.similarity_search(
# #             semantic_fallback,
# #             k=10,
# #             filter={"compliance_id": compliance_id}
# #         )

# #     if not docs:
# #         return "Not found in provided documents."

# #     context = "\n".join(d.page_content for d in docs)

# #     prompt = build_prompt(context, question)
# #     llm = get_llm()

# #     answer = llm.invoke(prompt).content.strip()

# #     return answer if answer else "Not found in provided documents."





# # 2.
# # def query_blocks(store, question, compliance_id):
# #     if store is None:
# #         return "No unstructured documents available for this compliance."

# #     # 1️⃣ Retrieve candidates (semantic search)
# #     docs = store.similarity_search(
# #         question,
# #         k=15,
# #         filter={"compliance_id": compliance_id}
# #     )

# #     if not docs:
# #         return "Not found in provided documents."

# #     # 2️⃣ RERANK using LLM (lightweight scoring)
# #     llm = get_llm()

# #     scored_docs = []

# #     for d in docs:
# #         score_prompt = f"""
# # You are scoring relevance.

# # Question:
# # {question}

# # Chunk:
# # {d.page_content}

# # Give a relevance score from 0 to 1.
# # Only output a number.
# # """
# #         try:
# #             score_text = llm.invoke(score_prompt).content.strip()
# #             score = float(score_text)
# #         except:
# #             score = 0.0

# #         scored_docs.append((score, d))

# #     # 3️⃣ Sort by relevance score
# #     scored_docs.sort(key=lambda x: x[0], reverse=True)

# #     # 4️⃣ Confidence thresholds
# #     HIGH_CONF = 0.6
# #     MED_CONF = 0.35

# #     top_score = scored_docs[0][0]

# #     # 5️⃣ Select chunks based on confidence
# #     if top_score >= HIGH_CONF:
# #         selected = [d.page_content for s, d in scored_docs[:4]]
# #         confidence_mode = "HIGH"
# #     elif top_score >= MED_CONF:
# #         selected = [d.page_content for s, d in scored_docs[:3]]
# #         confidence_mode = "MEDIUM"
# #     else:
# #         return "Not found in provided documents."

# #     context = "\n".join(selected)

# #     # 6️⃣ Confidence-aware prompt
# #     if confidence_mode == "HIGH":
# #         prompt = build_prompt(context, question)
# #     else:
# #         prompt = f"""
# # You are a compliance assistant.

# # The exact answer may not be explicitly stated.
# # Use the context to provide the closest relevant information.
# # Clearly mention that the exact wording is not present.

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """

# #     answer = llm.invoke(prompt).content.strip()

# #     return answer if answer else "Not found in provided documents."



# def query_blocks(store, question, compliance_id):
#     if store is None:
#         return "No unstructured documents available for this compliance."

#     # 1️⃣ Retrieve more candidates (semantic search)
#     docs = store.similarity_search(
#         question,
#         k=20,
#         filter={"compliance_id": compliance_id}
#     )

#     if not docs:
#         return "Not found in provided documents."

#     llm = get_llm()
#     scored = []

#     # 2️⃣ Rerank chunks by relevance
#     for d in docs:
#         score_prompt = f"""
# Score how relevant this chunk is to the question.

# Question:
# {question}

# Chunk:
# {d.page_content}

# Score from 0 to 1.
# Only output a number.
# """
#         try:
#             score = float(llm.invoke(score_prompt).content.strip())
#         except:
#             score = 0.0

#         scored.append((score, d))

#     # 3️⃣ Sort by relevance
#     scored.sort(key=lambda x: x[0], reverse=True)

#     top_score = scored[0][0]

#     # 4️⃣ Confidence thresholds
#     HIGH = 0.6
#     MEDIUM = 0.35

#     if top_score < MEDIUM:
#         return "Not found in provided documents."

#     # 5️⃣ Select best unique chunks
#     selected_chunks = []
#     seen = set()

#     for score, doc in scored:
#         text = doc.page_content.strip()
#         if text not in seen:
#             selected_chunks.append(text)
#             seen.add(text)
#         if len(selected_chunks) == 4:
#             break

#     context = "\n".join(selected_chunks)

#     # 6️⃣ Confidence-aware prompting
#     if top_score >= HIGH:
#         prompt = build_prompt(context, question)
#     else:
#         prompt = f"""
# You are a compliance assistant.

# The exact answer is not explicitly stated.
# Use the context to provide the closest relevant information.
# Clearly state this limitation.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     answer = llm.invoke(prompt).content.strip()
#     return answer if answer else "Not found in provided documents."




#######################################. eventfallback 
# import os
# from collections import OrderedDict
# from typing import List

# from vectorstore.faiss_store import create_vector_store, load_vector_store
# from processing.embedder import get_embedder
# from rag.prompt import build_prompt


# BASE_CHUNK_DIR = "data/chunks"
# BASE_EMBED_DIR = "data/embeddings"


# # ---------------------------------------------------------
# # INGESTION (UNCHANGED BEHAVIOR – SAFE)
# # ---------------------------------------------------------
# def ingest_blocks(blocks, compliance_id):
#     embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"

#     if os.path.exists(f"{embed_path}/index.faiss"):
#         print("✔ Loading existing embeddings (no OpenAI cost)")
#         return load_vector_store(embed_path)

#     print("⏳ Creating embeddings...")

#     all_chunks = []
#     all_metadata = []

#     for block in blocks:
#         if not block["type"]:
#             continue

#         block_type = block["type"].upper()
#         block_id = block["block_id"]

#         # Structured files are handled elsewhere
#         if block_type in ["XLS", "CSV", "GSHEET"]:
#             continue

#         text = block.get("extracted_text", "")
#         chunks = block.get("chunks", [])

#         for c in chunks:
#             all_chunks.append(c)
#             all_metadata.append({
#                 "compliance_id": compliance_id,
#                 "block_id": block_id,
#                 "source": block.get("value")
#             })

#     if not all_chunks:
#         return None

#     embedder = get_embedder()
#     store = create_vector_store(all_chunks, all_metadata, embedder)

#     os.makedirs(embed_path, exist_ok=True)
#     store.save_local(embed_path)

#     print("✅ Embeddings saved locally")
#     return store


# # ---------------------------------------------------------
# # 🔥 SEMANTIC + CONFIDENCE-AWARE QUERYING
# # ---------------------------------------------------------
# def query_blocks(store, question: str, compliance_id: str) -> str:
#     """
#     Semantic RAG with:
#     - closest-information fallback
#     - list completeness
#     - confidence awareness
#     - no duplicate answers
#     """

#     if store is None:
#         return "No unstructured documents available."

#     # 🔹 1. SEMANTIC SEARCH (NOT KEYWORD MATCH)
#     docs = store.similarity_search(
#         question,
#         k=8,
#         filter={"compliance_id": compliance_id}
#     )

#     if not docs:
#         return "Not found in provided documents."

#     # 🔹 2. MERGE + CLEAN CONTEXT
#     context_chunks = []
#     seen = set()

#     for d in docs:
#         text = d.page_content.strip()
#         if text and text not in seen:
#             seen.add(text)
#             context_chunks.append(text)

#     context = "\n".join(context_chunks)

#     # 🔹 3. CONFIDENCE HEURISTIC
#     # If keywords overlap but exact phrasing differs → closest answer mode
#     q_tokens = set(question.lower().split())
#     ctx_tokens = set(context.lower().split())

#     overlap_ratio = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)

#     # Thresholds chosen conservatively
#     if overlap_ratio < 0.08:
#         return "Not found in provided documents."

#     # 🔹 4. PROMPT-BASED SYNTHESIS (STRICT BUT HELPFUL)
#     prompt = build_prompt(context, question)

#     llm = get_llm()
#     answer = llm.invoke(prompt).content.strip()

#     # 🔹 5. FINAL SAFETY NET
#     if not answer or answer.lower().startswith("not found"):
#         return (
#             "The exact information is not explicitly mentioned in the documents. "
#             "However, closely related information is available:\n\n"
#             + context_chunks[0]
#         )

#     return answer


# # ---------------------------------------------------------
# # LAZY IMPORT (avoids circular import issues)
# # ---------------------------------------------------------
# def get_llm():
#     from processing.llm import get_llm as _get_llm
#     return _get_llm()








#.  pretty close version but no chunk saved 
# import os
# from typing import List

# from extractors.universal_extractor import extract_text
# from processing.chunker import chunk_text
# from vectorstore.faiss_store import create_vector_store, load_vector_store
# from processing.embedder import get_embedder
# from rag.prompt import build_prompt



# BASE_EMBED_DIR = "data/embeddings"


# # =====================================================
# # INGESTION (UNCHANGED – SAFE)
# # =====================================================
# def ingest_blocks(blocks, compliance_id):
#     embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"

#     if os.path.exists(f"{embed_path}/index.faiss"):
#         print("✔ Loading existing embeddings (no OpenAI cost)")
#         return load_vector_store(embed_path)

#     print("⏳ Creating embeddings...")

#     all_chunks = []
#     all_metadata = []

#     for block in blocks:
#         if not block.get("type"):
#             continue

#         block_type = block["type"].upper()
#         source = block["value"]

#         # 🚫 Skip structured here
#         if block_type in ["XLS", "CSV", "GSHEET"]:
#             continue

#         # ✅ EXTRACT TEXT
#         try:
#             text = extract_text(source)
#         except Exception as e:
#             print(f"⚠ Failed to extract {source}: {e}")
#             continue

#         if not text.strip():
#             continue

#         # ✅ CHUNK TEXT
#         chunks = chunk_text(text)

#         for c in chunks:
#             all_chunks.append(c)
#             all_metadata.append({
#                 "compliance_id": compliance_id,
#                 "source": source
#             })

#     if not all_chunks:
#         print("ℹ No unstructured documents found.")
#         return None

#     store = create_vector_store(
#         all_chunks,
#         all_metadata,
#         get_embedder()
#     )

#     os.makedirs(embed_path, exist_ok=True)
#     store.save_local(embed_path)

#     print("✅ Embeddings saved locally")
#     return store


# # =====================================================
# # 🔥 SEMANTIC + CONFIDENCE-AWARE RAG
# # =====================================================
# def query_blocks(store, question: str, compliance_id: str) -> str:
#     """
#     Robust semantic RAG with:
#     - semantic similarity (not keyword)
#     - closest-information fallback
#     - controlled output size
#     - no hallucination
#     """

#     if store is None:
#         return "No unstructured documents available."

#     # 1️⃣ Semantic retrieval
#     docs = store.similarity_search(
#         question,
#         k=8,
#         filter={"compliance_id": compliance_id}
#     )

#     if not docs:
#         return "Not found in provided documents."

#     # 2️⃣ Deduplicate + keep only most relevant text
#     seen = set()
#     context_chunks: List[str] = []

#     for d in docs:
#         text = d.page_content.strip()
#         if text and text not in seen:
#             seen.add(text)
#             context_chunks.append(text)
#         if len(context_chunks) == 4:  # hard limit
#             break

#     context = "\n".join(context_chunks)

#     # 3️⃣ Confidence heuristic (semantic overlap)
#     q_tokens = set(question.lower().split())
#     ctx_tokens = set(context.lower().split())

#     overlap = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)

#     if overlap < 0.07:
#         return "Not found in provided documents."

#     # 4️⃣ Ask LLM with STRICT rules
#     llm = _get_llm()
#     prompt = build_prompt(context, question)
#     answer = llm.invoke(prompt).content.strip()

#     # 5️⃣ Controlled fallback (closest info only)
#     if not answer or answer.lower().startswith("not found"):
#         return (
#             "The exact information is not explicitly mentioned in the documents. "
#             "However, closely related information is available:\n\n"
#             + context_chunks[0]
#         )

#     return answer


# # =====================================================
# # Lazy import (prevents circular imports)
# # =====================================================
# def _get_llm():
#     from processing.llm import get_llm
#     return get_llm()










#chunks saved 
# import os
# from typing import List

# from extractors.universal_extractor import extract_text
# from processing.chunker import chunk_text
# from vectorstore.faiss_store import create_vector_store, load_vector_store
# from processing.embedder import get_embedder
# from rag.prompt import build_prompt


# BASE_EMBED_DIR = "data/embeddings"
# BASE_CHUNK_DIR = "data/chunks"


# # =====================================================
# # INGESTION (FIXED – CHUNKS ARE SAVED)
# # =====================================================
# def ingest_blocks(blocks, compliance_id):
#     embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"
#     chunk_path = f"{BASE_CHUNK_DIR}/compliance_{compliance_id}"

#     # If embeddings already exist, load them
#     if os.path.exists(f"{embed_path}/index.faiss"):
#         print("✔ Loading existing embeddings (no OpenAI cost)")
#         return load_vector_store(embed_path)

#     print("⏳ Creating embeddings...")

#     os.makedirs(chunk_path, exist_ok=True)

#     all_chunks = []
#     all_metadata = []

#     for block in blocks:
#         if not block.get("type"):
#             continue

#         block_type = block["type"].upper()
#         source = block["value"]

#         # 🚫 Skip structured files here
#         if block_type in ["XLS", "CSV", "GSHEET"]:
#             continue

#         # ✅ Extract text
#         try:
#             text = extract_text(source)
#         except Exception as e:
#             print(f"⚠ Failed to extract {source}: {e}")
#             continue

#         if not text.strip():
#             print(f"⚠ Empty text extracted from {source}")
#             continue

#         # ✅ Chunk text
#         chunks = chunk_text(text)

#         if not chunks:
#             continue

#         # ✅ Save chunks to disk (FOR DEBUGGING & TRACEABILITY)
#         safe_name = os.path.basename(source).replace("/", "_").replace(" ", "_")
#         chunk_file = os.path.join(chunk_path, f"{safe_name}.txt")

#         with open(chunk_file, "w", encoding="utf-8") as f:
#             for i, chunk in enumerate(chunks, start=1):
#                 f.write(f"[CHUNK {i}]\n")
#                 f.write(chunk.strip())
#                 f.write("\n\n" + "=" * 80 + "\n\n")

#         # ✅ Prepare for embeddings
#         for c in chunks:
#             all_chunks.append(c)
#             all_metadata.append({
#                 "compliance_id": compliance_id,
#                 "source": source
#             })

#     if not all_chunks:
#         print("ℹ No unstructured documents found.")
#         return None

#     store = create_vector_store(
#         all_chunks,
#         all_metadata,
#         get_embedder()
#     )

#     os.makedirs(embed_path, exist_ok=True)
#     store.save_local(embed_path)

#     print("✅ Embeddings saved locally")
#     print(f"📄 Chunks saved at: {chunk_path}")

#     return store


# # =====================================================
# # SEMANTIC + CONFIDENCE-AWARE RAG (UNCHANGED)
# # =====================================================
# def query_blocks(store, question: str, compliance_id: str) -> str:
#     if store is None:
#         return "No unstructured documents available."

#     docs = store.similarity_search(
#         question,
#         k=8,
#         filter={"compliance_id": compliance_id}
#     )

#     if not docs:
#         return "Not found in provided documents."

#     seen = set()
#     context_chunks: List[str] = []

#     for d in docs:
#         text = d.page_content.strip()
#         if text and text not in seen:
#             seen.add(text)
#             context_chunks.append(text)
#         if len(context_chunks) == 4:
#             break

#     context = "\n".join(context_chunks)

#     q_tokens = set(question.lower().split())
#     ctx_tokens = set(context.lower().split())
#     overlap = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)

#     if overlap < 0.07:
#         return "Not found in provided documents."

#     llm = _get_llm()
#     prompt = build_prompt(context, question)
#     answer = llm.invoke(prompt).content.strip()

#     if not answer or answer.lower().startswith("not found"):
#         return (
#             "The exact information is not explicitly mentioned in the documents. "
#             "However, closely related information is available:\n\n"
#             + context_chunks[0]
#         )

#     return answer


# # =====================================================
# # Lazy import (prevents circular imports)
# # =====================================================
# def _get_llm():
#     from processing.llm import get_llm
#     return get_llm()







import os
import re
from typing import List

from extractors.universal_extractor import extract_text
from processing.chunker import chunk_text
from vectorstore.faiss_store import create_vector_store, load_vector_store
from processing.embedder import get_embedder
from rag.prompt import build_prompt

BASE_EMBED_DIR = "data/embeddings"
BASE_CHUNK_DIR = "data/chunks"


# =====================================================
# INGESTION (SAFE + TRACEABLE)
# =====================================================
def ingest_blocks(blocks, compliance_id):
    embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"
    chunk_path = f"{BASE_CHUNK_DIR}/compliance_{compliance_id}"

    if os.path.exists(f"{embed_path}/index.faiss"):
        print("✔ Loading existing embeddings (no OpenAI cost)")
        return load_vector_store(embed_path)

    print("⏳ Creating embeddings...")

    os.makedirs(chunk_path, exist_ok=True)

    all_chunks = []
    all_metadata = []

    for block in blocks:
        if not block.get("type"):
            continue

        block_type = block["type"].upper()
        source = block["value"]

        if block_type in ["XLS", "CSV", "GSHEET"]:
            continue

        try:
            text = extract_text(source)
        except Exception as e:
            print(f"⚠ Failed to extract {source}: {e}")
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        # Save chunks for inspection
        safe_name = os.path.basename(source).replace(" ", "_")
        with open(
            os.path.join(chunk_path, f"{safe_name}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for i, c in enumerate(chunks, 1):
                f.write(f"[CHUNK {i}]\n{c}\n\n{'='*80}\n\n")

        for c in chunks:
            all_chunks.append(c)
            all_metadata.append({
                "compliance_id": compliance_id,
                "source": source
            })

    if not all_chunks:
        print("ℹ No unstructured documents found.")
        return None

    store = create_vector_store(
        all_chunks,
        all_metadata,
        get_embedder()
    )

    os.makedirs(embed_path, exist_ok=True)
    store.save_local(embed_path)

    print("✅ Embeddings saved locally")
    print(f"📄 Chunks saved at: {chunk_path}")
    return store


# =====================================================
# 🔥 ENTITY-FIRST SEMANTIC RAG (FIXED)
# =====================================================
def query_blocks(store, question: str, compliance_id: str) -> str:
    if store is None:
        return "No unstructured documents available."

    # 1️⃣ Semantic retrieval
    docs = store.similarity_search(
        question,
        k=12,
        filter={"compliance_id": compliance_id}
    )

    if not docs:
        return "Not found in provided documents."

    # 2️⃣ Extract key entities from question
    entities = _extract_entities(question)

    # 3️⃣ Entity-first filtering
    entity_matched = []
    for d in docs:
        text = d.page_content.lower()
        if any(e in text for e in entities):
            entity_matched.append(d)

    # Fallback if entity not explicitly found
    if entity_matched:
        docs = entity_matched

    # 4️⃣ Build focused context
    context_chunks: List[str] = []
    seen = set()

    for d in docs:
        text = d.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            context_chunks.append(text)
        if len(context_chunks) == 3:
            break

    if not context_chunks:
        return "Not found in provided documents."

    context = "\n\n".join(context_chunks)

    # 5️⃣ Ask LLM with STRICT prompt
    llm = _get_llm()
    prompt = build_prompt(context, question)
    answer = llm.invoke(prompt).content.strip()

    # 6️⃣ Controlled fallback
    if not answer or answer.lower().startswith("not found"):
        return (
            "The exact information is not explicitly mentioned in the documents. "
            "However, closely related information is available:\n\n"
            + context_chunks[0]
        )

    return answer


# =====================================================
# UTILITIES
# =====================================================
def _extract_entities(question: str) -> List[str]:
    """
    Lightweight entity extractor:
    - Acronyms
    - Capitalized terms
    - Keywords longer than 3 chars
    """
    q = question.lower()

    acronyms = re.findall(r"\b[A-Z]{2,}\b", question)
    keywords = re.findall(r"\b[a-z]{4,}\b", q)

    entities = set(a.lower() for a in acronyms)
    entities.update(keywords)

    return list(entities)


def _get_llm():
    from processing.llm import get_llm
    return get_llm()
