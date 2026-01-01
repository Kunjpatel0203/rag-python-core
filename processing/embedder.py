from langchain_openai import OpenAIEmbeddings

def get_embedder():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
