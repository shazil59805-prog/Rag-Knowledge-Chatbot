import streamlit as st
import faiss, pickle, os
from sentence_transformers import SentenceTransformer

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
# Page Config & Dark Theme CSS Injection
# ---------------------------
st.set_page_config(page_title="RAG Intelligence Hub", layout="wide")

# Force Premium Dark/Black Mode via CSS Styling
st.markdown("""
    <style>
    /* Entire Main Background to Deep Cyber Black */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    
    /* Left Sidebar Custom Dark Theme */
    section[data-testid="stSidebar"] {
        background-color: #060913 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Text Fields Formatting */
    .stTextInput>div>div>input {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }
    
    /* Global Text Visibility Force Color */
    h1, h2, h3, h4, p, label, span {
        color: #f8fafc !important;
    }
    
    /* Expander Internal Text Contrast */
    .stMarkdown div div p {
        color: #cbd5e1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar UI (Clean & Modern)
# ---------------------------
st.sidebar.markdown("<h2 style='color: #1e3a8a; font-weight:700; margin-bottom:20px;'>⚙️ Core Engine Settings</h2>", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Select System Intelligence Model:",
    ["Direct (local engine)", "Gemini (Google AI)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-weight:600; font-size:14px; margin-bottom:5px;'>📊 Search Fine-Tuning</p>", unsafe_allow_html=True)

n_pages = st.sidebar.slider(
    "Documents retrieved per query",
    min_value=1, max_value=4, value=2
)

max_chars = st.sidebar.slider(
    "Max visible character buffer",
    min_value=100, max_value=1500, value=1000
)

# ---------------------------
# Main Dashboard UI (Aapka Bilkul Original Text)
# ---------------------------
st.markdown("<div class='main-title'>🌐 AI & PDC Intelligence Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Retrieval-Augmented Generation (RAG) System — Developed by Shazil (CCP Project)</div>", unsafe_allow_html=True)

# Dynamic Badge Display (As per your design)
if "Direct"
