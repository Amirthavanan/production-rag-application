import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pickle
import os
from app.chunker import chunk_text
from app.embedder import create_embeddings
from app.vector_store import build_faiss_index
from app.bm25_retriever import build_bm25
from app.retriever import hybrid_retrieve
from app.llm import generate_answer

# Page Configuration
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG PDF Assistant")
st.markdown("Upload your PDF document(s), ask questions, and get precise answers powered by Hybrid Retrieval & LLM.")

# Sidebar for PDF Upload
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Document(s)", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Upload one or more PDF files to analyze."
)

def process_uploaded_pdfs(files):
    """Extract text, chunk, embed, and index uploaded PDF files directly from memory."""
    all_chunks = []
    
    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "source": uploaded_file.name,
                        "page": i + 1
                    })
    return all_chunks

def build_pipeline(chunks):
    """Build FAISS index and BM25 retriever from chunks."""
    embeddings = np.array(create_embeddings(chunks))
    index = build_faiss_index(embeddings)
    bm25, tokenized_corpus = build_bm25(chunks)
    return index, bm25, tokenized_corpus

# Handle indexing based on upload or pre-existing index
if uploaded_files:
    # Check if we need to re-index new uploads
    upload_names = [f.name for f in uploaded_files]
    if "current_files" not in st.session_state or st.session_state["current_files"] != upload_names:
        with st.spinner("Processing uploaded PDF(s) and building index..."):
            chunks = process_uploaded_pdfs(uploaded_files)
            if not chunks:
                st.error("No extractable text found in the uploaded PDF(s). Please try another document.")
                st.stop()
            
            index, bm25, tokenized_corpus = build_pipeline(chunks)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["bm25"] = bm25
            st.session_state["tokenized_corpus"] = tokenized_corpus
            st.session_state["current_files"] = upload_names
            st.sidebar.success(f"✅ Indexed {len(uploaded_files)} PDF(s) ({len(chunks)} chunks)")

else:
    # Fallback to pre-built index or local sample PDF if available
    if "chunks" not in st.session_state:
        if os.path.exists("chunks.pkl") and os.path.exists("faiss.index"):
            with st.spinner("Loading default document index..."):
                with open("chunks.pkl", "rb") as f:
                    chunks = pickle.load(f)
                index = fitz.faiss.read_index("faiss.index") if hasattr(fitz, "faiss") else None
                import faiss
                index = faiss.read_index("faiss.index")
                bm25, tokenized_corpus = build_bm25(chunks)
                
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["bm25"] = bm25
                st.session_state["tokenized_corpus"] = tokenized_corpus
                st.session_state["current_files"] = ["Default Document"]
        else:
            st.info("👆 Please upload a PDF file in the sidebar to get started.")

# Display document info in sidebar
if "current_files" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Active Document Stats")
    st.sidebar.write(f"**Files:** {', '.join(st.session_state['current_files'])}")
    st.sidebar.write(f"**Total Chunks:** {len(st.session_state.get('chunks', []))}")

# Main Q&A Interface
if "chunks" in st.session_state and st.session_state["chunks"]:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Enter your question based on the uploaded document(s):", placeholder="e.g. What are the key skills listed in the document?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_btn = st.button("🔍 Ask Question", use_container_width=True)

    if submit_btn or query and False:
        if not query.strip():
            st.warning("Please enter a question before clicking Ask.")
        else:
            with st.spinner("Searching document & generating answer..."):
                chunks = st.session_state["chunks"]
                index = st.session_state["index"]
                bm25 = st.session_state["bm25"]
                tokenized_corpus = st.session_state["tokenized_corpus"]
                
                results = hybrid_retrieve(query, index, chunks, bm25, tokenized_corpus)
                data = generate_answer(query, results)

                st.markdown("---")
                st.markdown("### 🤖 Answer")
                if isinstance(data, dict):
                    st.write(data.get("answer", "No answer generated."))
                    
                    citations = data.get("citations", [])
                    if citations:
                        st.markdown("#### 📌 Citations")
                        for citation in citations:
                            st.info(f"📄 **Source:** `{citation.get('source', 'Uploaded PDF')}` | **Page:** `{citation.get('page', 'N/A')}`")
                else:
                    st.write(str(data))

                st.markdown("---")
                with st.expander("📖 View Retrieved Context Chunks"):
                    for i, context in enumerate(results):
                        st.markdown(f"**Chunk {i+1}** *(Source: {context.get('source', 'PDF')}, Page {context.get('page', 'N/A')})*")
                        st.text_area(f"Chunk text #{i+1}", context["text"], height=100, disabled=True)
