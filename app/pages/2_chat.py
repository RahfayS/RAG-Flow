from app.db.chat_history_manager import ChatHistoryManager
from utils.initialize_session_state import initialize_session_state
import streamlit as st
from pathlib import Path
import uuid
def status_line(label: str, ok: bool) -> str:
    color = "#4a8a4a" if ok else "#4a4840"
    dot = "●" if ok else "○"
    return (
        f'<div style="font-family:var(--font-mono);font-size:0.63rem;'
        f'letter-spacing:0.1em;color:{color};margin-bottom:0.35rem">'
        f'{dot} {label}</div>'
    )


if len(st.session_state) == 0:
    st.write("Session state is empty")
    initialize_session_state()


def ready_to_chat():
    """Verifies if minimal chatting requirements are met before starting a chat"""
    if not st.session_state.llm and not st.session_state.embedding_model:
        initialize_session_state()
    

chm = ChatHistoryManager()
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatting",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(path: str) -> None:
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css("assets/styles.css")

# ==== Sidebar ====
with st.sidebar:
    model_loaded = st.session_state.get("llm") is not None
    embed_loaded = st.session_state.get("embedding_model") is not None
    reranker_loaded = st.session_state.get("reranker") is not None

    st.markdown(
        status_line("MODEL LOADED", model_loaded) +
        status_line("EMBEDDINGS LOADED",embed_loaded) +
        status_line(f"RE-RANKER",reranker_loaded),
        unsafe_allow_html=True,
    )
    st.markdown("---")



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
             Start Chatting!<br>
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
    )


# ==== Sidebar =====
with st.sidebar:
    st.markdown("---")
    st.caption("RAG Workflow v0.1.0")

# ==== Stats ====
st.markdown(
    """
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Previous Chats</div>
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

if not ready_to_chat():
    st.write("Please configure LLM and Embedding Model to begin chatting")

st.markdown('<hr class="divider">', unsafe_allow_html=True)




if ready_to_chat():
    
    if "session_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    if st.sidebar.button("Start Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    st.markdown("---")

    st.sidebar.markdown("Previous Chats")

    for chat_id in chm.get_all_chats():
        if st.sidebar.button(chat_id[:5]):
             st.session_state.session_id = chat_id
             st.session_state.chat_history = chm.load_message_history(chat_id)


    chat_id = st.session_state.chat_id