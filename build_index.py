from app.ingestion import process_pdfs
from app.embedder import create_embeddings
from app.vector_store import build_faiss_index
import numpy as np
import pickle
import faiss

print("Processing PDFs...")

chunks = process_pdfs("data/raw")

print("Creating embeddings...")
embeddings = create_embeddings(chunks)
embeddings = np.array(embeddings)

print("Building FAISS index...")
index = build_faiss_index(embeddings)

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Save FAISS index
faiss.write_index(index, "faiss.index")

print("✅ Done! Files created:")
print("chunks.pkl")
print("faiss.index")