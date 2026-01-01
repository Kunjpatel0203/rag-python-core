from langchain_community.vectorstores import FAISS

def create_vector_store(texts, metadatas, embedder):
    return FAISS.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas
    )

def load_vector_store(path):
    from processing.embedder import get_embedder
    return FAISS.load_local(
        path,
        get_embedder(),
        allow_dangerous_deserialization=True
    )
