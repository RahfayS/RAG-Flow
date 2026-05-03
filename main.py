from config import SESSIONS_DIR

from pathlib import Path
import streamlit as st
from pathlib import Path
import uuid
import os


# ==== Define Page Config ====
st.set_page_config(
    page_title="RAG Workflow",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==== Session-state defaults ====
defaults = {
    "llm": None,
    "embedding_model": None,
    "text_splitter": None,
    "reranker": None,
    "temperature": 0.2,
    "max_tokens": 512,
    "top_k": 5,
    "chunk_size": 256,
    "chunk_overlap": 32,
    "cohere_api_key": "",
    "openai_api_key": "",
    "session_id": uuid.uuid4(),
    "chat_id": uuid.uuid4(),
    "chat_history": []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

def initialize_user():
    session_id = st.session_state.session_id

    docs_path = os.path.join(SESSIONS_DIR,"documents")
    vector_path = os.path.join(SESSIONS_DIR,"vector_store")
    chat_path = os.path.join(SESSIONS_DIR,"chat")
    metadata_path = f"{SESSIONS_DIR}/metadata.json"


# ==== Load CSS Styles ====
def load_css(path: str) -> None:
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css("assets/styles.css")


# ==== SideBar ====
with st.sidebar:
    st.caption("RAG Workflow v0.1.0")
    st.write("Sidebar")


# ==== Hero ====
st.markdown(
    """
    <div class="hero-container">
        <div class="status-badge">
            <div class="status-dot"></div>
            SYSTEM READY
        </div>
        <div class="hero-eyebrow">Retrieval-Augmented Generation</div>
        <h1 class="hero-title">
            Ask questions.<br>
            Get <em>grounded</em> answers.
        </h1>
        <p class="hero-subtitle">
            A modular RAG pipeline that ingests your documents, indexes them
            into a vector store, and retrieves precise context to power
            accurate, citation-backed responses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==== Stats ====
st.markdown(
    """
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Documents indexed</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Chunks stored</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Queries run</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">—</div>
            <div class="stat-label">Avg latency</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ==== Define Columns ====
col_left, col_right = st.columns([3, 2], gap="large")

# --- Left Column ---
with col_left:
    st.markdown('<p class="section-header">Pipeline Architecture</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pipeline-grid">
            <div class="pipeline-step">
                <div class="step-index">01</div>
                <span class="step-icon">📥</span>
                <div class="step-name">Ingest</div>
                <div class="step-desc">Upload PDFs, Markdown, or plain text files for processing.</div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">02</div>
                <span class="step-icon">✂️</span>
                <div class="step-name">Chunk</div>
                <div class="step-desc">Split documents into overlapping semantic chunks.</div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">03</div>
                <span class="step-icon">🔢</span>
                <div class="step-name">Embed</div>
                <div class="step-desc">Convert chunks into dense vector representations.</div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">04</div>
                <span class="step-icon">🗄️</span>
                <div class="step-name">Store</div>
                <div class="step-desc">Persist embeddings in a vector database index.</div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">05</div>
                <span class="step-icon">🔍</span>
                <div class="step-name">Retrieve</div>
                <div class="step-desc">Similarity search returns the most relevant chunks.</div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">06</div>
                <span class="step-icon">💬</span>
                <div class="step-name">Generate</div>
                <div class="step-desc">LLM synthesises a grounded, cited answer.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Right Column ---
with col_right:
    st.markdown('<p class="section-header">Features</p>', unsafe_allow_html=True)
    features = [
        ("Configurable chunking",   "Adjustable size & overlap"),
        ("Pluggable embeddings",    "OpenAI, Cohere, local models"),
        ("Vector store agnostic",   "FAISS, Pinecone, Qdrant, Chroma"),
        ("Source citations",        "Every answer links to its chunks"),
        ("Streaming responses",     "Token-by-token display"),
        ("Evaluation harness",      "RAGAS metrics out of the box"),
    ]
    items_html = "".join(
        f"""
        <div class="feature-item">
            <div class="feature-dot"></div>
            <div class="feature-text">
                <strong>{title}</strong><br>
                <span>{desc}</span>
            </div>
        </div>
        """
        for title, desc in features
    )
    st.markdown(items_html, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="section-header">Get Started</p>', unsafe_allow_html=True)


# ==== Quick Start ====
with st.expander("Quick-start: run locally"):
    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. 

# 3. Launch the app
streamlit run app.py
        """,
        language="bash",
    )