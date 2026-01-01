import os

# ===== Existing loaders (DO NOT REMOVE) =====
from loaders.url_loader import load_url
from loaders.file_loader import download_file

# ===== New browser-based loader (ADDED) =====
from loaders.url_loader_browser import load_url_with_browser

from extractors.universal_extractor import extract_text
from extractors.excel_metadata_extractor import extract_excel_metadata

from processing.chunker import chunk_text
from processing.embedder import get_embedder
from processing.llm import get_llm

from vectorstore.faiss_store import create_vector_store, load_vector_store
from rag.prompt import build_prompt
from structured.store import save_structured_file


# =============================
# 🔧 CONFIG
# =============================
USE_BROWSER = True   # ✅ Toggle safely (True = Playwright, False = old loader)

BASE_CHUNK_DIR = "data/chunks"
BASE_EMBED_DIR = "data/embeddings"


# =============================
# 🧩 Helper: Save Chunks
# =============================
def save_chunks(compliance_id, block_id, source_name, chunks):
    os.makedirs(f"{BASE_CHUNK_DIR}/compliance_{compliance_id}", exist_ok=True)

    file_path = (
        f"{BASE_CHUNK_DIR}/compliance_{compliance_id}/"
        f"block_{block_id}_{source_name}.txt"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"COMPLIANCE ID: {compliance_id}\n")
        f.write(f"BLOCK ID: {block_id}\n")
        f.write(f"SOURCE: {source_name}\n")
        f.write("=" * 50 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            f.write(f"[CHUNK {i}]\n{chunk}\n\n")


# =============================
# 🧠 INGEST BLOCKS
# =============================
def ingest_blocks(blocks, compliance_id):
    embed_path = f"{BASE_EMBED_DIR}/compliance_{compliance_id}"

    # ✅ Reuse embeddings if present
    if os.path.exists(f"{embed_path}/index.faiss"):
        print("✔ Loading existing embeddings (no OpenAI cost)")
        return load_vector_store(embed_path)

    print("⏳ Creating embeddings...")

    all_chunks = []
    all_metadata = []

    for block in blocks:
        if not block.get("type"):
            continue

        block_type = block["type"].upper()
        block_id = block["block_id"]
        value = block["value"]

        # ==================================================
        # 📊 STRUCTURED FILES (EXCEL / CSV / GSHEET)
        # ==================================================
        if block_type in ["XLS", "XLSX", "CSV", "GSHEET"]:
            # 1️⃣ Save for NLQ (structured)
            save_structured_file(block, compliance_id)

            # 2️⃣ Extract metadata / merged headers → RAG
            try:
                metadata_text = extract_excel_metadata(value)
            except Exception as e:
                print("[EXCEL METADATA ERROR]", e)
                continue

            if metadata_text and metadata_text.strip():
                meta_chunks = chunk_text(metadata_text)
                source_name = (
                    os.path.basename(value).replace(".", "_") + "_metadata"
                )

                save_chunks(compliance_id, block_id, source_name, meta_chunks)

                for c in meta_chunks:
                    all_chunks.append(c)
                    all_metadata.append({
                        "compliance_id": compliance_id,
                        "block_id": block_id,
                        "source": value,
                        "type": "excel_metadata"
                    })

            # 🚫 DO NOT fall through to unstructured handling
            continue

        # ==================================================
        # 🌐 URL INGESTION (SAFE HYBRID)
        # ==================================================
        if block_type == "URL":
            if USE_BROWSER:
                page_text, downloaded_docs = load_url_with_browser(value)
            else:
                page_text, downloaded_docs = load_url(value)

            # Page text → chunks
            page_chunks = chunk_text(page_text)
            save_chunks(compliance_id, block_id, "url_page", page_chunks)

            for c in page_chunks:
                all_chunks.append(c)
                all_metadata.append({
                    "compliance_id": compliance_id,
                    "block_id": block_id,
                    "source": value,
                    "type": "url_page"
                })

            # Downloaded docs (PDF / XLS / PPT etc.)
            for path in downloaded_docs:
                try:
                    text = extract_text(path)
                except Exception as e:
                    print("[DOC EXTRACT ERROR]", e)
                    continue

                doc_chunks = chunk_text(text)
                name = os.path.basename(path).replace(".", "_")

                save_chunks(compliance_id, block_id, name, doc_chunks)

                for c in doc_chunks:
                    all_chunks.append(c)
                    all_metadata.append({
                        "compliance_id": compliance_id,
                        "block_id": block_id,
                        "source": path,
                        "type": "url_document"
                    })

            continue

        # ==================================================
        # 📄 PDF / DOC / PPT (UNSTRUCTURED)
        # ==================================================
        try:
            text = extract_text(value)
        except Exception as e:
            print("[FILE EXTRACT ERROR]", e)
            continue

        chunks = chunk_text(text)
        name = os.path.basename(value).replace(".", "_")

        save_chunks(compliance_id, block_id, name, chunks)

        for c in chunks:
            all_chunks.append(c)
            all_metadata.append({
                "compliance_id": compliance_id,
                "block_id": block_id,
                "source": value,
                "type": "unstructured"
            })

    # ==================================================
    # 🚫 No RAG content
    # ==================================================
    if not all_chunks:
        print("ℹ No unstructured documents found. Skipping vector store creation.")
        return None

    # ==================================================
    # 🧠 Create Vector Store
    # ==================================================
    embedder = get_embedder()
    store = create_vector_store(all_chunks, all_metadata, embedder)

    os.makedirs(embed_path, exist_ok=True)
    store.save_local(embed_path)

    print("✅ Embeddings saved locally")
    return store


# =============================
# 🔍 QUERY UNSTRUCTURED (RAG)
# =============================
def query_blocks(store, question, compliance_id):
    if store is None:
        return "No unstructured documents available for this compliance."

    # 1️⃣ Primary retrieval
    docs = store.similarity_search(
        question,
        k=10,
        filter={"compliance_id": compliance_id}
    )

    # 2️⃣ If nothing found, try semantic fallback
    if not docs:
        semantic_fallback = f"{question} related information"
        docs = store.similarity_search(
            semantic_fallback,
            k=10,
            filter={"compliance_id": compliance_id}
        )

    if not docs:
        return "Not found in provided documents."

    context = "\n".join(d.page_content for d in docs)

    prompt = build_prompt(context, question)
    llm = get_llm()

    answer = llm.invoke(prompt).content.strip()

    return answer if answer else "Not found in provided documents."
