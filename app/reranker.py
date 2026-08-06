"""Cross-encoder reranking module with fallback handling."""
try:
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception:
    _reranker = None


def rerank(query, chunks, top_k=5):
    """Rerank chunks using CrossEncoder model, fallback to top_k if model fails."""
    if not chunks:
        return []
    if _reranker is None:
        return chunks[:top_k]

    try:
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        scores = _reranker.predict(pairs)
        scored_chunks = list(zip(chunks, scores))
        ranked = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_k]]
    except Exception:
        return chunks[:top_k]