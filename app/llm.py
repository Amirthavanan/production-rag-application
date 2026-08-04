import os
import json
import streamlit as st

def get_api_keys():
    """Retrieve API keys from Streamlit secrets or environment variables."""
    groq_key = None
    openai_key = None

    # Check Streamlit secrets first
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_key = st.secrets["GROQ_API_KEY"]
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Fallback to environment variables
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")

    return groq_key, openai_key

def generate_answer(query, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
[Chunk {i+1}]
Source: {chunk.get('source', 'Document')}
Page: {chunk.get('page', 'N/A')}
Text: {chunk.get('text', '')}
"""

    prompt = f"""
You are a strict AI assistant answering questions based on the provided document chunks.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT make up information.
3. If the answer is not in the context, state "I couldn't find the answer in the provided document."
4. Include citations (source filename and page number).
5. Return ONLY valid JSON format.

Output Format:
{{
  "answer": "Your detailed answer here.",
  "citations": [
    {{
      "source": "filename.pdf",
      "page": 1
    }}
  ]
}}

Context:
{context}

Question:
{query}
"""

    groq_key, openai_key = get_api_keys()

    if not groq_key and not openai_key:
        return {
            "answer": "⚠️ **API Key Missing**: Please set `GROQ_API_KEY` (or `OPENAI_API_KEY`) in your **Streamlit App Settings -> Secrets** or in your `.env` file.\n\n"
                      "**How to fix on Streamlit Cloud:**\n"
                      "1. Open your app on Streamlit Cloud.\n"
                      "2. Click **Settings** (⚙️) -> **Secrets**.\n"
                      "3. Add: `GROQ_API_KEY = \"gsk_your_groq_api_key_here\"` (or `OPENAI_API_KEY = \"sk-...\"`)\n"
                      "4. Save and re-run your app.",
            "citations": []
        }

    # Try Groq API first
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            raw_output = response.choices[0].message.content
            return json.loads(raw_output)
        except Exception as e:
            err_msg = str(e)
            if "Connection error" in err_msg or "APIConnectionError" in err_msg:
                err_msg += " (Check your network connection or verify that GROQ_API_KEY is valid)."
            if not openai_key:
                return {
                    "answer": f"⚠️ **Groq API Error**: {err_msg}",
                    "citations": []
                }

    # Fallback to OpenAI API if available
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o-mini",
                response_format={"type": "json_object"}
            )
            raw_output = response.choices[0].message.content
            return json.loads(raw_output)
        except Exception as e:
            return {
                "answer": f"⚠️ **OpenAI API Error**: {str(e)}",
                "citations": []
            }

    return {
        "answer": "⚠️ Unable to generate response with available API keys.",
        "citations": []
    }
