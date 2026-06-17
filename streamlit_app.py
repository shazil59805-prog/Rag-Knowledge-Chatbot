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
# Page Config & Complete Dark Theme CSS
# ---------------------------
st.set_page_config(page_title="PDC HPC Portal", layout="wide")

# Injection of Professional High-Tech Dark Theme
st.markdown("""
    <style>
    /* Main Screen Background */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    
    /* Left Sidebar Custom Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #070a13 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Input Field Styling */
    .stTextInput>div>div>input {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    
    /* Get Answer Button Custom CSS */
    .stButton>button {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #0284c7 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
    }

    /* Dynamic Badges styling */
    .badge-local {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #0284c7;
    }
    .badge-gemini {
        background-color: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #7e22ce;
    }

    /* Force all text headers to align properly on Dark Mode */
    h1, h2, h3, h4, p, label, span {
        color: #f8fafc !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar (Settings UI)
# ---------------------------
st.sidebar.markdown("<h2 style='color: #38bdf8; font-weight:700; font-size:22px; margin-top:10px;'>⚙️ Settings Center</h2>", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Operation mode",
    ["Direct (local engine)", "Gemini (Google AI)"],
    index=0
)

st.sidebar.markdown("<br><hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

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
# Main Page Title Area
# ---------------------------
st.title("🌐 PDC High-Performance Computing Portal")

# Premium Status Badges instead of simple text layout
if "Direct" in mode:
    st.markdown("<span class='badge-local'>⚙️ System Pipeline: Local FAISS Vector Store Active</span>", unsafe_allow_html=True)
else:
    st.markdown("<span class='badge-gemini'>🔮 System Pipeline: Cloud Gemini Engine Active</span>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        if mode.startswith("Direct") and index is not None:
            # Local FAISS search
            q_emb = model.encode([query], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=n_pages)

            st.markdown("<h3 style='color:#38bdf8; font-weight:700; margin-top:15px;'>📖 Local Context Retrieved</h3>", unsafe_allow_html=True)
            for rank, idx in enumerate(I[0], 1):
                with st.expander(f"🔍 Relevant Reference Source {rank}"):
                    full_text = chunks[idx]
                    # Dynamic responsive text wrapper inside the expander
                    st.markdown(f"<div style='font-size:15px; line-height:1.7; color:#cbd5e1;'>{full_text}</div>", unsafe_allow_html=True)
                
        elif mode.startswith("Gemini"):
            # ✅ Direct Gemini API call with 2 answers + links
            try:
                api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in secrets.")
                else:
                    genai.configure(api_key=api_key)
                    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                    # multiple answers (2 candidates) + request for links
                    resp = gemini_model.generate_content(
                        f"{query}\n\nPlease also provide source links or references if possible.",
                        generation_config={"candidate_count": 2}
                    )

                    st.markdown("<h3 style='color:#c084fc; font-weight:700; margin-top:15px;'>🤖 Gemini Answers</h3>", unsafe_allow_html=True)
                    for i, cand in enumerate(resp.candidates, 1):
                        if cand.content.parts:
                            answer = cand.content.parts[0].text
                            st.markdown(f"#### 🧠 Answer Formulation Choice {i}")
                            st.markdown(f"<div style='background-color:#1e293b; padding:15px; border-radius:8px; border-left:4px solid #a855f7; color:#e2e8f0; margin-bottom:15px;'>{answer}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error calling Gemini API: {e}")

        else:
            st.error("⚠️ Local Engine not available. Build the index first.")

# ---------------------------
# Notes Section
# ---------------------------
st.markdown("<br><hr style='border-color:#1e293b;'>", unsafe_allow_html=True)
st.info(
    """
    **Notes**
    - The first run may download model weights (internet required).  
    - The answers are extractive: taken from retrieved documents or Wikipedia-style content.  
    - Gemini mode uses Google Gemini API. Make sure `GEMINI_API_KEY` is set in Streamlit secrets.  
    - In Gemini mode, answers may also include references or links.  
    """
)
