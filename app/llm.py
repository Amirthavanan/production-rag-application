import requests
import json


def generate_answer(query, retrieved_chunks):
    context = ""

    # Build context
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
[Chunk {i+1}]
Source: {chunk['source']}
Page: {chunk['page']}
Text: {chunk['text']}
"""

    # Prompt
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

    # Call Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    # Convert response safely
    try:
        data = response.json()

        print("Answer generated successfully")

        # Handle Ollama errors
        if "response" not in data:
            return {
                "answer": f"Ollama error: {data}",
                "citations": []
            }

        raw_output = data["response"]

        # Try parsing JSON from model output
        try:
            parsed = json.loads(raw_output)

            # Handle nested JSON string
            if (
                isinstance(parsed.get("answer"), str)
                and parsed["answer"].strip().startswith("{")
            ):
                parsed = json.loads(parsed["answer"])

            return parsed

        except Exception as parse_error:
            return {
                "answer": raw_output,
                "citations": [],
                "error": str(parse_error)
            }

    except Exception as response_error:
        return {
            "answer": "Failed to get response from Ollama",
            "citations": [],
            "error": str(response_error)
        }