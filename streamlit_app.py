import streamlit as st
import faiss, pickle, os
from sentence_transformers import SentenceTransformer
import os
if not os.path.exists("index"):
    import build_index
    build_index.build_index(data_dir="data", index_dir="index")
# ✅ Gemini import
import google.generativeai as genai  

# ---------------------------
# Load local FAISS index (safe loader)
# ---------------------------
INDEX_DIR = "index"
chunks, meta, index = [], [], None

if os.path.exists(os.path.join(INDEX_DIR, "faiss.index")):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)

    # Robust loader: handle both dict and list
    if isinstance(data, dict):
        chunks = data.get("chunks", [])
        meta = data.get("meta", [])
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], str):
            chunks = data
            meta = [{} for _ in chunks]
        elif len(data) > 0 and isinstance(data[0], dict):
            if "text" in data[0]:
                chunks = [d.get("text", "") for d in data]
                meta = [d.get("meta", {}) for d in data]
            elif "chunk" in data[0]:
                chunks = [d.get("chunk", "") for d in data]
                meta = [d.get("meta", {}) for d in data]
            else:
                chunks = [str(d) for d in data]
                meta = [{} for _ in chunks]
        else:
            chunks = [str(d) for d in data]
            meta = [{} for _ in chunks]

# ---------------------------
# Model load
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Knowledge Chatbot — CCP Project", layout="wide")

# ---------------------------
# Sidebar (Settings)
# ---------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Operation mode",
    ["Direct (local engine)", "Gemini (Google AI)"],
    index=0
)

# Adjusted ranges
n_pages = st.sidebar.slider(
    "📄 Number of Wikipedia pages (1–4)",
    min_value=1,
    max_value=4,
    value=2
)

max_chars = st.sidebar.slider(
    "🔤 Max characters per page (100–1500)",
    min_value=100,
    max_value=1500,
    value=500
)

# ---------------------------
# Main Page
# ---------------------------
st.title("🌐 Knowledge Chatbot — By Shazil CCP Project")
st.markdown(f"**Mode:** {mode}")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        if mode.startswith("Direct") and index is not None:
            # Local FAISS search
            q_emb = model.encode([query], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=n_pages)

            st.subheader("📖 Local Engine Results")
            for rank, idx in enumerate(I[0], 1):
                st.markdown(f"**Result {rank}:**")
                st.write(chunks[idx][:max_chars] + ("..." if len(chunks[idx]) > max_chars else ""))
                st.markdown("---")

        elif mode.startswith("Gemini"):
            # ✅ Direct Gemini API call with 2 answers + links
            try:
                api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in secrets.")
                else:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    # multiple answers (2 candidates) + request for links
                    resp = model.generate_content(
                        f"{query}\n\nPlease also provide source links or references if possible.",
                        generation_config={"candidate_count": 2}
                    )

                    st.subheader("🤖 Gemini Answers")
                    for i, cand in enumerate(resp.candidates, 1):
                        if cand.content.parts:
                            answer = cand.content.parts[0].text
                            st.markdown(f"**Answer {i}:** {answer}")
                            st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error calling Gemini API: {e}")

        else:
            st.error("⚠️ Local Engine not available. Build the index first.")

# ---------------------------
# Notes Section
# ---------------------------
st.info(
    """
    **Notes**
    - The first run may download model weights (internet required).  
    - The answers are extractive: taken from retrieved documents or Wikipedia-style content.  
    - Gemini mode uses Google Gemini API. Make sure `GEMINI_API_KEY` is set in Streamlit secrets.  
    - In Gemini mode, answers may also include references or links.  
    """
)
