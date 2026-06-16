import os
from PyPDF2 import PdfReader

# Document loader
def load_documents(data_dir):
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append(f.read())
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)
    return docs

# Simple chunking (split into small parts for embeddings)
def chunk_documents(docs, chunk_size=500, overlap=50):
    chunks = []
    for doc in docs:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunks.append(doc[start:end])
            start = end - overlap
    return chunks
