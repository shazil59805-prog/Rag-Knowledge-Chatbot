import os
import pickle
import faiss
from utils import load_documents, chunk_documents
from sentence_transformers import SentenceTransformer

def build_index(data_dir, index_dir):
    # Load & chunk documents
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents")
    print(f"Total chunks: {len(chunks)}")

    # Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    # Save metadata
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index written")
    print("Metadata written")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    build_index("data", "index")
build_index()
