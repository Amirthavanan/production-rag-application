# Production RAG Application

A production-style Retrieval-Augmented Generation (RAG) system built using:

- FastAPI
- Streamlit
- FAISS
- BM25
- Ollama
- Hybrid Retrieval
- PDF Ingestion

## Features

- Hybrid retrieval (BM25 + vector search)
- Citation-based responses
- Local LLM inference using Ollama
- Streamlit frontend
- FastAPI backend
- Evaluation pipeline

## Run Backend

```bash
python -m uvicorn app.main:app

##Run Frontend

```bash
streamlit run frontend.py


