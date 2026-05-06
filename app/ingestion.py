import os
from app.pdf_reader import extract_text_from_pdf
from app.chunker import chunk_text

def process_pdfs(folder_path):
    all_chunks = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            pages = extract_text_from_pdf(file_path)

            for page in pages:
                chunks = chunk_text(page["text"])

                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "source": file,
                        "page": page["page_number"]
                    })

    return all_chunks