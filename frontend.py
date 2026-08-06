"""
AI Resume Screening & HR Document Q&A System - Streamlit Frontend.
Powered by Hybrid Retrieval (FAISS + BM25), Cross-Encoder Reranking, and LLM Grounding.
"""
import os
import streamlit as st

from app.llm import generate_answer, get_api_keys
from app.pipeline import build_pipeline, load_default_index, process_resume_pdfs
from app.retriever import hybrid_retrieve

# ------------------------------------------------------------------------------
# 1. Page Configuration & Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & HR Document Q&A",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern HR Workspace styling
st.markdown(
    """
    <style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2b46 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .badge-container {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border-color: rgba(168, 85, 247, 0.3);
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border-color: rgba(34, 197, 94, 0.3);
    }

    /* Stat Cards */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-val {
        color: #38bdf8;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .stat-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result Cards & Citation Chips */
    .answer-card {
        background: #0f172a;
        border-left: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .citation-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. Domain Presets & System Prompts
# ------------------------------------------------------------------------------
RESUME_QUICK_ACTIONS = {
    "📋 Candidate Executive Summary": "Provide a concise professional summary of this candidate's background, core competencies, and career progression.",
    "🎯 Technical & Soft Skills Matrix": "Extract all technical tools, programming languages, methodologies, and soft skills listed in the candidate's resume.",
    "⏳ Work History & Experience Timeline": "List each employment position, employer name, duration, and key accomplishments in chronological order.",
    "🎓 Education & Professional Certifications": "Summarize all academic degrees, educational institutions, honors, and professional certifications mentioned.",
    "💪 Core Strengths & Key Highlights": "Identify the top 3-5 major strengths or standout achievements demonstrated in the resume.",
    "🏆 Fit Score & Hiring Recommendation": "Evaluate the candidate's overall profile, give a score from 0-100, and provide a Hire / Hold / Reject recommendation with justification based strictly on the resume.",
    "❓ Tailored Interview Questions": "Formulate 5 specific technical and behavioral interview questions tailored to verify claims made in this resume.",
    "🚩 Red Flags & Resume Gap Analysis": "Highlight any potential red flags, unaccounted employment gaps, overlapping dates, or vague job responsibility descriptions.",
}

HR_POLICY_QUICK_ACTIONS = {
    "🌴 Paid Time Off & Leave Policy": "What are the company policies regarding annual leave, sick leave, maternity/paternity leave, and PTO carry-over?",
    "💻 Remote Work & Flexible Hours": "What are the rules and guidelines governing remote work, work-from-home allowances, and core work hours?",
    "💰 Salary, Compensation & Reviews": "Explain the performance review cycle, bonus structure, salary review timelines, and promotion criteria.",
    "🏥 Employee Benefits & Health Insurance": "Summarize the health insurance coverage, wellness perks, medical reimbursement, and retirement plan benefits.",
    "📜 Code of Conduct & Compliance": "What are the key workplace conduct rules, anti-harassment policies, data security requirements, and NDA guidelines?",
    "🚀 Onboarding & Probation Period": "Describe the onboarding process, probation period duration, evaluation criteria, and confirmation process for new hires.",
}

RESUME_SCREENING_PREFIX = (
    "You are an expert HR Talent Acquisition Specialist performing candidate screening. "
    "Analyze the provided resume context thoroughly. Answer using ONLY the resume content provided. "
    "Where information is missing, state 'Not mentioned in the resume'."
)

HR_POLICY_PREFIX = (
    "You are an expert HR Policy Assistant. "
    "Answer employee and management questions using ONLY the official HR policy context provided. "
    "If the topic is not covered in the document, state 'Not specified in the uploaded HR policy documents'."
)

# ------------------------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------------------------
def init_session_state():
    """Ensure all required session state variables exist."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "current_files" not in st.session_state:
        st.session_state["current_files"] = []
    if "target_jd" not in st.session_state:
        st.session_state["target_jd"] = ""


def run_rag_query(query: str, system_prefix: str = ""):
    """Execute hybrid search + LLM generation and record history."""
    if "chunks" not in st.session_state or not st.session_state["chunks"]:
        st.warning("⚠️ No documents indexed yet. Please upload PDF files in the sidebar.")
        return None, None

    full_query = f"{system_prefix}\n\n{query}" if system_prefix else query

    results = hybrid_retrieve(
        full_query,
        st.session_state["index"],
        st.session_state["chunks"],
        st.session_state["bm25"],
        st.session_state["tokenized_corpus"],
    )

    data = generate_answer(full_query, results)

    # Append to interaction history
    st.session_state["history"].append({
        "query": query,
        "answer": data.get("answer", "No answer generated."),
        "citations": data.get("citations", []),
        "results": results,
    })

    return data, results


def render_answer_block(data: dict, results: list):
    """Render structured AI Answer, Source Citations, and Context Evidence."""
    st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Screening & HR Analysis Answer")
    st.markdown(data.get("answer", "No answer generated."))
    st.markdown("</div>", unsafe_allow_html=True)

    citations = data.get("citations", [])
    if citations:
        st.markdown("#### 📌 Grounded Source Citations")
        citation_html = ""
        for citation in citations:
            src = citation.get("source", "Document PDF")
            pg = citation.get("page", "N/A")
            citation_html += f"<div class='citation-chip'>📄 <b>{src}</b> | Page {pg}</div>"
        st.markdown(citation_html, unsafe_allow_html=True)

    if results:
        with st.expander("📖 View Retrieved Document Chunks (Evidence Context)"):
            for i, chunk in enumerate(results):
                st.markdown(
                    f"**Chunk #{i+1}** — *Source: `{chunk.get('source')}` | Page `{chunk.get('page')}`*"
                )
                st.text_area(
                    f"Content Chunk #{i+1}",
                    value=chunk.get("text", ""),
                    height=100,
                    key=f"chunk_text_{i}_{str(abs(hash(chunk.get('text', ''))))[:8]}",
                    disabled=True,
                )


# ------------------------------------------------------------------------------
# 4. Main Application Layout
# ------------------------------------------------------------------------------
def main():
    init_session_state()

    # --- Hero Header ---
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">💼 AI Resume Screening & HR Document Q&A</h1>
            <div class="hero-subtitle">
                Enterprise Production RAG Workspace for Candidate Evaluation, Job Matching & HR Policy Intelligence.
            </div>
            <div class="badge-container">
                <span class="badge">⚡ Hybrid RAG (FAISS + BM25)</span>
                <span class="badge badge-purple">🎯 Cross-Encoder Reranked</span>
                <span class="badge badge-green">🔒 Strict Grounding & Citations</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Workspace Settings")

        domain_mode = st.radio(
            "📌 Primary Document Mode",
            options=["👤 Candidate Resumes", "📚 HR Policies & Handbooks"],
            help="Select the domain mode for document analysis.",
        )

        st.markdown("---")
        st.subheader("📁 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload candidate PDF resumes or company HR policy documents.",
        )

        # Ingestion Processing
        if uploaded_files:
            upload_names = [f.name for f in uploaded_files]
            if st.session_state.get("current_files") != upload_names:
                with st.spinner("⚡ Extracting text, chunking & building Hybrid Index..."):
                    chunks = process_resume_pdfs(uploaded_files)
                    if not chunks:
                        st.error("No extractable text found. Please upload valid text-based PDFs.")
                    else:
                        index, bm25, tokenized_corpus = build_pipeline(chunks)
                        st.session_state.update(
                            {
                                "chunks": chunks,
                                "index": index,
                                "bm25": bm25,
                                "tokenized_corpus": tokenized_corpus,
                                "current_files": upload_names,
                            }
                        )
                        st.success(f"✅ Indexed {len(uploaded_files)} file(s) ({len(chunks)} chunks)")

        elif "chunks" not in st.session_state:
            # Fallback to prebuilt index if available
            default_index = load_default_index()
            if default_index:
                chunks, index, bm25, tokenized_corpus = default_index
                st.session_state.update(
                    {
                        "chunks": chunks,
                        "index": index,
                        "bm25": bm25,
                        "tokenized_corpus": tokenized_corpus,
                        "current_files": ["Pre-indexed Default Knowledge Base"],
                    }
                )
                st.info("💡 Loaded pre-built FAISS & BM25 index.")

        st.markdown("---")
        st.subheader("🤖 AI Engine Status")
        groq_key, openai_key = get_api_keys()

        if groq_key:
            st.success("🟢 Connected to Groq LLM (Llama 3.3 70B)")
        elif openai_key:
            st.info("🟢 Connected to OpenAI (GPT-4o mini)")
        else:
            st.warning("⚠️ No API Key found in `.env` or environment")

        st.markdown("---")
        st.subheader("🎯 Job Description (JD) Benchmark")
        target_jd_input = st.text_area(
            "Paste Target Job Description (Optional)",
            value=st.session_state.get("target_jd", ""),
            placeholder="Paste role requirements here to compare candidate resumes against specific skill requirements...",
            height=120,
        )
        st.session_state["target_jd"] = target_jd_input

        if st.button("🗑️ Reset Workspace & Index", use_container_width=True):
            for k in ["chunks", "index", "bm25", "tokenized_corpus", "current_files", "history"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # --- Top Metrics Bar ---
    num_files = len(st.session_state.get("current_files", []))
    num_chunks = len(st.session_state.get("chunks", []))
    active_files = ", ".join(st.session_state.get("current_files", ["None"]))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val'>{num_files}</div><div class='stat-lbl'>Documents Loaded</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val'>{num_chunks}</div><div class='stat-lbl'>RAG Chunks Indexed</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val'>Hybrid</div><div class='stat-lbl'>FAISS + BM25 Vector Engine</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        status_txt = "Ready" if num_chunks > 0 else "Upload PDFs"
        st.markdown(
            f"<div class='stat-card'><div class='stat-val'>{status_txt}</div><div class='stat-lbl'>System Status</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if num_chunks == 0:
        st.info(
            "👈 **Getting Started:** Upload one or more Candidate Resumes or HR Policy PDFs in the sidebar to begin RAG screening & Q&A."
        )
        return

    # --- Workflow Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👤 Candidate Resume Screening",
            "⚖️ Job Fit & JD Matcher",
            "📚 HR Policy Q&A",
            "📜 Screening Session Log",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: Candidate Resume Screening
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("🔍 Deep Candidate Resume Analysis")
        st.caption(f"Currently screening document(s): `{active_files}`")

        c1, c2 = st.columns([1, 1])
        with c1:
            selected_action = st.selectbox(
                "⚡ Quick HR Screening Preset",
                options=list(RESUME_QUICK_ACTIONS.keys()),
            )
        with c2:
            custom_resume_q = st.text_input(
                "Or enter custom screening query",
                placeholder="e.g. Does the candidate have experience with Python, FastAPI, and AWS deployment?",
            )

        if st.button("🚀 Run Screening Analysis", key="btn_screen", use_container_width=True):
            query_to_run = (
                custom_resume_q.strip()
                if custom_resume_q.strip()
                else RESUME_QUICK_ACTIONS[selected_action]
            )

            with st.spinner("🔎 Performing Hybrid Retrieval & Cross-Encoder Reranking..."):
                data, results = run_rag_query(query_to_run, system_prefix=RESUME_SCREENING_PREFIX)

            if data:
                render_answer_block(data, results)

    # --------------------------------------------------------------------------
    # TAB 2: Job Description Fit Analysis
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("⚖️ Candidate Profile vs Job Description Match")

        jd_text = st.session_state.get("target_jd", "").strip()
        if not jd_text:
            st.info("💡 Paste a Job Description in the sidebar under 'Target Job Description Benchmark' or enter it below to evaluate fit.")

        user_jd = st.text_area(
            "Target Job Description / Key Requirements",
            value=jd_text,
            height=140,
            placeholder="e.g. Senior Software Engineer with 4+ years Python experience, RAG system design, PyTorch, Vector databases (FAISS/Pinecone), Docker, and REST APIs...",
        )

        col_fit1, col_fit2, col_fit3 = st.columns(3)
        with col_fit1:
            btn_fit_score = st.button("📊 Evaluate Candidate Fit Score", use_container_width=True)
        with col_fit2:
            btn_gap_skills = st.button("⚠️ Missing Skills & Gaps", use_container_width=True)
        with col_fit3:
            btn_jd_questions = st.button("❓ JD-Specific Interview Qs", use_container_width=True)

        jd_query = None
        if btn_fit_score:
            jd_query = (
                f"Compare the candidate's resume against this Job Description:\n'''\n{user_jd}\n'''\n"
                f"Provide a 0-100 overall match score, breakdown of matching qualifications vs missing requirements, and final hiring recommendation."
            )
        elif btn_gap_skills:
            jd_query = (
                f"Compare the candidate's resume against this Job Description:\n'''\n{user_jd}\n'''\n"
                f"List all required skills, experience levels, or certifications from the JD that are MISSING or NOT clearly demonstrated in the resume."
            )
        elif btn_jd_questions:
            jd_query = (
                f"Based on the candidate's resume and this Target Job Description:\n'''\n{user_jd}\n'''\n"
                f"Generate 5 targeted technical interview questions to assess whether the candidate meets the core JD requirements."
            )

        if jd_query:
            if not user_jd.strip():
                st.error("Please enter a Job Description above to run the match analysis.")
            else:
                with st.spinner("⚖️ Benchmarking candidate resume against Job Description..."):
                    data, results = run_rag_query(jd_query, system_prefix=RESUME_SCREENING_PREFIX)
                if data:
                    render_answer_block(data, results)

    # --------------------------------------------------------------------------
    # TAB 3: HR Policy Q&A
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📚 HR Document & Policy Assistant")
        st.caption("Ask questions about company policies, employee benefits, onboarding, or workplace guidelines.")

        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            selected_policy_preset = st.selectbox(
                "⚡ HR Policy Presets",
                options=list(HR_POLICY_QUICK_ACTIONS.keys()),
            )
        with p_col2:
            custom_policy_q = st.text_input(
                "Or ask a custom HR policy question",
                placeholder="e.g. What is the process for submitting travel expense reimbursements?",
            )

        if st.button("🔍 Search HR Policy Documents", key="btn_policy", use_container_width=True):
            policy_query = (
                custom_policy_q.strip()
                if custom_policy_q.strip()
                else HR_POLICY_QUICK_ACTIONS[selected_policy_preset]
            )

            with st.spinner("📖 Searching policy documents & retrieving grounded answer..."):
                data, results = run_rag_query(policy_query, system_prefix=HR_POLICY_PREFIX)

            if data:
                render_answer_block(data, results)

    # --------------------------------------------------------------------------
    # TAB 4: Screening Session Log
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("📜 Screening & Q&A Interaction Log")
        history = st.session_state.get("history", [])

        if not history:
            st.info("No queries executed in this session yet. Run a screening analysis or HR question to record history.")
        else:
            st.write(f"**Total Queries in Session:** {len(history)}")
            if st.button("🧹 Clear History Log"):
                st.session_state["history"] = []
                st.rerun()

            for idx, item in enumerate(reversed(history)):
                with st.expander(f"Q#{len(history)-idx}: {item['query'][:80]}...", expanded=(idx == 0)):
                    st.markdown(f"**Question:** {item['query']}")
                    st.markdown(f"**Answer:**\n{item['answer']}")

                    citations = item.get("citations", [])
                    if citations:
                        st.markdown("**Citations:**")
                        for c in citations:
                            st.caption(f"• `{c.get('source')}` (Page {c.get('page')})")


if __name__ == "__main__":
    main()
