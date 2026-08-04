import streamlit as st
import faiss
import pickle
from app.bm25_retriever import build_bm25
from app.retriever import hybrid_retrieve
from app.llm import generate_answer

st.set_page_config(page_title="Ask My Resume", page_icon="📄")
st.title("📄 Resume Bot")

import os
import numpy as np

# Cache heavy data loading
@st.cache_resource
def load_rag_components():
    if not (os.path.exists("chunks.pkl") and os.path.exists("faiss.index")):
        st.info("Index files missing. Generating FAISS index and chunks from PDFs...")
        from app.ingestion import process_pdfs
        from app.embedder import create_embeddings
        from app.vector_store import build_faiss_index

        pdf_folder = "data/raw" if os.path.exists("data/raw") else "."
        chunks = process_pdfs(pdf_folder)
        embeddings = np.array(create_embeddings(chunks))
        index = build_faiss_index(embeddings)

        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        faiss.write_index(index, "faiss.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss.index")
    bm25, tokenized_corpus = build_bm25(chunks)
    return chunks, index, bm25, tokenized_corpus

chunks, index, bm25, tokenized_corpus = load_rag_components()

query = st.text_input("Enter your question")

if st.button("Get"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Generating answer..."):
            results = hybrid_retrieve(query, index, chunks, bm25, tokenized_corpus)
            data = generate_answer(query, results)

            st.subheader("Answer")
            if isinstance(data, dict):
                st.write(data.get("answer", "No answer generated."))
                
                citations = data.get("citations", [])
                if citations:
                    st.subheader("Citations")
                    for citation in citations:
                        st.write(f"📄 {citation.get('source', 'Unknown')} | Page {citation.get('page', 'N/A')}")
            else:
                st.write(str(data))

            st.subheader("Retrieved Context")
            for i, context in enumerate(results):
                with st.expander(f"Chunk {i+1}"):
                    st.write(context["text"])
