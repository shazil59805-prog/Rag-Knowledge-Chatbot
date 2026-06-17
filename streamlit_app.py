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
# Page Config & Custom Styling
# ---------------------------
st.set_page_config(page_title="RAG Intelligence Hub", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #4b5563;
        margin-bottom: 25px;
    }
    .mode-badge-local {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #bae6fd;
    }
    .mode-badge-gemini {
        background-color: #f3e8ff;
        color: #6b21a8;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #e9d5ff;
    }
    .result-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 15px;
        border-left: 5px solid #2563eb;
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
# Main Dashboard UI
# ---------------------------
st.markdown("<div class='main-title'>🌐 PDC Distributed Architecture Knowledge Base</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Retrieval-Augmented Generation (RAG) System — Developed by Shazil (CCP Project)</div>", unsafe_allow_html=True)

# Dynamic Badge Display
if "Direct" in mode:
    st.markdown("<span class='mode-badge-local'>⚙️ System Mode: Independent Local FAISS Vector Search</span>", unsafe_allow_html=True)
else:
    st.markdown("<span class='mode-badge-gemini'>✨ System Mode: Cloud-Augmented Generative AI (Gemini Engine)</span>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Query Input Field
query = st.text_input("📝 Ask your research or technical question:", placeholder="e.g., Explain Amdahl's Law vs Gustafson's Law or CNN architecture...")

# Process Button with modern placement
col1, col2 = st.columns([1, 5])
with col1:
    submit_btn = st.button("🚀 Analyze Query", use_container_width=True)

if submit_btn:
    if query.strip() == "":
        st.warning("⚠️ Query input field cannot be empty. Please type a valid question.")
    else:
        if mode.startswith("Direct") and index is not None:
            # Local FAISS search
            q_emb = model.encode([query], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=n_pages)

            st.markdown("<h3 style='color:#1e3a8a; font-weight:700; margin-top:20px;'>📖 Retained Contextual Knowledge Base</h3>", unsafe_allow_html=True)
            
            for rank, idx in enumerate(I[0], 1):
                with st.expander(f"📂 Verified Reference Source {rank} (FAISS Vector Segment)"):
                    full_text = chunks[idx]
                    st.markdown(f"<div style='font-size:15px; line-height:1.7; color:#334155; padding: 10px 5px;'>{full_text}</div>", unsafe_allow_html=True)
                    
        elif mode.startswith("Gemini"):
            # ✅ Direct Gemini API call with 2 answers + links
            try:
                api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("❌ Gemini API key config missing. Please set GEMINI_API_KEY in Streamlit cloud secrets.")
                else:
                    genai.configure(api_key=api_key)
                    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

                    resp = gemini_model.generate_content(
                        f"{query}\n\nPlease also provide source links or references if possible.",
                        generation_config={"candidate_count": 2}
                    )

                    st.markdown("<h3 style='color:#6b21a8; font-weight:700; margin-top:20px;'>🤖 Generative AI Synthesis (Dual Hypotheses)</h3>", unsafe_allow_html=True)
                    
                    for i, cand in enumerate(resp.candidates, 1):
                        if cand.content.parts:
                            answer = cand.content.parts[0].text
                            st.markdown(f"#### 🧠 Cognitive Response Formulation {i}")
                            st.markdown(f"<div style='background-color:#f8fafc; padding:15px; border-radius:8px; border-left:4px solid #8b5cf6;'>{answer}</div>", unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Core Gateway Timeout / Gemini API Error: {e}")

        else:
            st.error("⚠️ Database Offline. Local vector storage cluster could not be loaded.")

# ---------------------------
# Footer Project Notes
# ---------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info(
        """
        **💡 Architectural Framework Notes:**
        - **Local Vector Engine:** Utilizes an `all-MiniLM-L6-v2` Sentence Transformer mapped to a FAISS index layer.
        - **Generative AI Pipeline:** Maps dynamic vector segments directly to Google's fast generative infrastructure.
        """
    )
