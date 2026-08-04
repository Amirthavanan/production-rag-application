import streamlit as st
import faiss
import pickle
from app.bm25_retriever import build_bm25
from app.retriever import hybrid_retrieve
from app.llm import generate_answer

st.set_page_config(page_title="Ask My Resume", page_icon="📄")
st.title("📄 Resume Bot")

# Cache heavy data loading
@st.cache_resource
def load_rag_components():
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
