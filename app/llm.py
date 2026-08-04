import os
import json
import streamlit as st
from groq import Groq

# Get API key from Streamlit secrets or environment variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

def generate_answer(query, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
[Chunk {i+1}]
Source: {chunk['source']}
Page: {chunk['page']}
Text: {chunk['text']}
"""

    prompt = f"""
You are a strict assistant.

Rules:
1. Answer ONLY from the provided context
2. Do NOT make up information
3. If answer is not found, say "I don't know"
4. ALWAYS include citations
5. Return ONLY valid JSON

Output Format:
{{
  "answer": "...",
  "citations": [
    {{
      "source": "...",
      "page": 1
    }}
  ]
}}

Context:
{context}

Question:
{query}
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
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
        return {
            "answer": f"Error generating response: {str(e)}",
            "citations": []
        }
