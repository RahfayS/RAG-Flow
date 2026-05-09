from utils.initialize_session_state import initialize_session_state
from config import SESSIONS_DIR

from pathlib import Path
import streamlit as st
import os


# ==== Define Page Config ====
st.set_page_config(
    page_title="RAG Workflow",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==== Session-state defaults ====
initialize_session_state()

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


st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Login ---
st.markdown(
    """
    <div class="login-wrap">
      <div class="login-card">
        <div class="login-eyebrow">Authentication</div>
        <div class="login-title">Sign in to<br>your workspace.</div>
        <p class="login-subtitle">
          Access your documents, retrieval history, and<br>
          saved pipeline configurations.
        </p>
    """,
    unsafe_allow_html=True,
)

if not st.user["is_logged_in"]:
    st.markdown('<div class="google-btn-wrap">', unsafe_allow_html=True)
    if st.button("Continue with Google", use_container_width=True):
        st.login("google")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Persist user + register in DB on first login
    if "user_added" not in st.session_state:
        st.session_state.user = st.user
        created = st.session_state.user_db.add_user(
            sub   = st.session_state.user["sub"],
            email = st.session_state.user["email"],
            name  = st.session_state.user["name"],
        )
        st.session_state.uid = st.session_state.user_db.get_uid(st.session_state.user["sub"])
        if created:
            st.session_state.user_added = True

    name = st.session_state.user.get("name", "User")
    email = st.session_state.user.get("email", "")

    st.markdown(
        f"""
        <div class="login-welcome">
          <div class="login-welcome-icon">✦</div>
          <div>
            <div class="login-welcome-name">{name}</div>
            <div class="login-welcome-sub">● Signed in · {email}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Close the card div
st.markdown("</div></div>", unsafe_allow_html=True)