from rank_bm25 import BM25Okapi

def build_bm25(chunks):
    tokenized_corpus = [chunk["text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus
def bm25_search(query, bm25, tokenized_corpus, chunks, top_k=10):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [chunks[i] for i in top_indices]
    return results