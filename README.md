
# Production RAG Application (Ask My Docs)

A production-style Retrieval-Augmented Generation (RAG) application built using FastAPI, Streamlit, Hybrid Retrieval, FAISS, BM25, Ollama, and evaluation pipelines.

---

# Features

- PDF document ingestion
- Semantic chunking
- Hybrid retrieval (BM25 + Vector Search)
- FAISS vector database
- Cross-encoder reranking
- Ollama local LLM inference
- Citation-based grounded responses
- Streamlit frontend
- Evaluation pipeline
- FastAPI backend API

---

# Architecture

<img src="Screenshots/RAG_Architecture.png">

---

# Tech Stack

- Python
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- BM25
- Ollama
- Llama3
- RAG Pipeline

---

# Project Structure

```bash
RAG_Project/
├── app/
├── evaluation/
├── data/
├── frontend.py
├── requirements.txt
└── README.md
```
## Screenshots:


## UI
<img src="Screenshots/UI.png">
## Fast-API
<img src="Screenshots/swagger.png">

## Run Backend

```bash
python -m uvicorn app.main:app
```

<<<<<<< HEAD



=======
## Run Frontend

```bash
streamlit run frontend.py
```

