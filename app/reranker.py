from sentence_transformers import CrossEncoder

# Load model once
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank(query, chunks, top_k=5):
    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker.predict(pairs)

    # Attach scores
    scored_chunks = list(zip(chunks, scores))

    # Sort by score descending
    ranked = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

    # Return top_k chunks
    return [item[0] for item in ranked[:top_k]]