"""Reusable document pipeline helpers for the RAG application."""
import os
import pickle

import faiss
import fitz  # PyMuPDF
import numpy as np

from app.bm25_retriever import build_bm25
from app.chunker import chunk_text
from app.embedder import create_embeddings
from app.vector_store import build_faiss_index


def process_resume_pdfs(files):
    """Extract text, chunk, embed, and index uploaded resume PDFs in memory."""
    chunks = []
    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                for chunk in chunk_text(text):
                    chunks.append({
                        "text": chunk,
                        "source": uploaded_file.name,
                        "page": i + 1,
                    })
    return chunks


def build_pipeline(chunks):
    """Build FAISS + BM25 retrieval stack from chunks."""
    embeddings = np.array(create_embeddings(chunks))
    index = build_faiss_index(embeddings)
    bm25, tokenized_corpus = build_bm25(chunks)
    return index, bm25, tokenized_corpus


def load_default_index():
    """Load prebuilt chunks.pkl + faiss.index if present, otherwise None."""
    if not (os.path.exists("chunks.pkl") and os.path.exists("faiss.index")):
        return None
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    bm25, tokenized_corpus = build_bm25(chunks)
    return chunks, index, bm25, tokenized_corpus
