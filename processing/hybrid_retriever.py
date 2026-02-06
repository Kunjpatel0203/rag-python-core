import os
import re
from collections import Counter
import math


# -----------------------------
# basic tokenizer
# -----------------------------
def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


# -----------------------------
# load all chunks from txt files
# -----------------------------
def load_all_chunks(chunk_dir):
    chunks = []

    for file in os.listdir(chunk_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(chunk_dir, file)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # split by [CHUNK x]
        parts = content.split("[CHUNK")
        for p in parts:
            if "]" in p:
                text = p.split("]", 1)[1].strip()
                if text:
                    chunks.append(text)

    return chunks


# -----------------------------
# BM25 scoring
# -----------------------------
def bm25_search(query, chunks, k=10):
    tokenized_docs = [tokenize(c) for c in chunks]
    doc_freq = Counter()
    for doc in tokenized_docs:
        for word in set(doc):
            doc_freq[word] += 1

    N = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(N, 1)

    q_tokens = tokenize(query)

    scores = []

    for i, doc in enumerate(tokenized_docs):
        score = 0
        dl = len(doc)

        for term in q_tokens:
            if term not in doc_freq:
                continue

            df = doc_freq[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            freq = doc.count(term)
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl)

            score += idf * ((freq * 2.5) / denom)

        if score > 0:
            scores.append((score, chunks[i]))

    scores.sort(reverse=True)
    return [text for _, text in scores[:k]]
