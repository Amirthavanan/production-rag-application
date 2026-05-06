from sentence_transformers import SentenceTransformer
import numpy as np
from app.reranker import rerank
from app.bm25_retriever import bm25_search

model = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query, index, chunks, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


def hybrid_retrieve(query, index, chunks, bm25, tokenized_corpus, top_k=5):
    vector_results = vector_search(query, index, chunks, top_k=15)
    bm25_results = bm25_search(query, bm25, tokenized_corpus, chunks, top_k=15)

    # Merge
    combined = vector_results + bm25_results

    # Remove duplicates
    seen = set()
    unique = []
    for item in combined:
        key = (item["source"], item["page"], item["text"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # 🔥 RERANK HERE
    reranked = rerank(query, unique, top_k=top_k)

    return reranked