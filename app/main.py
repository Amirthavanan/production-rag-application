from fastapi import FastAPI
from app.llm import generate_answer
import pickle
import faiss
from app.bm25_retriever import build_bm25
from app.retriever import hybrid_retrieve

app = FastAPI()
# Load prebuilt data
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index("faiss.index")

bm25, tokenized_corpus = build_bm25(chunks)

@app.post("/ask")
def ask_question(query: str):
    results = hybrid_retrieve(query, index, chunks, bm25, tokenized_corpus)

    response = generate_answer(query, results)

    return {
    "answer": response["answer"],
    "citations": response["citations"],
    "contexts": [chunk["text"] for chunk in results]
}