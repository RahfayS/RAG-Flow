from langchain_classic.text_splitter import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter

from db.model_preferences import ModelPreferencesDB

from utils.load_model import load_model,load_embedding_model, load_reranker_model
from utils.build_config import build_llm_config, build_embedding_config
from utils.initialize_session_state import initialize_session_state
from utils.pdf_utils import save_uploaded_files

from pathlib import Path
import streamlit as st


def status_line(label: str, ok: bool) -> str:
    color = "#4a8a4a" if ok else "#4a4840"
    dot = "●" if ok else "○"
    return (
        f'<div style="font-family:var(--font-mono);font-size:0.63rem;'
        f'letter-spacing:0.1em;color:{color};margin-bottom:0.35rem">'
        f'{dot} {label}</div>'
    )


if len(st.session_state) == 0:
    initialize_session_state()

# ==== Page Config ====
st.set_page_config(
    page_title="RAG Configuration",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(path: str) -> None:
    st.markdown(f"<style>{Path(path).read_text()}</style>", unsafe_allow_html=True)

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


#  ==== Hero ====
st.markdown(
    """
    <div class="hero-container">
        <div class="status-badge">
            <div class="status-dot"></div>
            CONFIGURATION
        </div>
        <div class="hero-eyebrow">Retrieval-Augmented Generation</div>
        <h1 class="hero-title">
            Configure your<br><em>pipeline.</em>
        </h1>
        <p class="hero-subtitle">
            Select a language model, tune retrieval parameters, and attach an
            embedding model before starting a session. Different choices
            significantly affect retrieval quality and latency.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ====  Stats row ====
llm = st.session_state.get("llm","Not Loaded")
embedding_model = st.session_state.get("embedding_model","Not Loaded")
reranker = st.session_state.get("reranker","Not Loaded")

st.markdown(
    f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">{status_line("Model Loaded",llm)}</div>
            <div class="stat-label">{f"Using {llm.model}" if llm is not None else "Not Loaded"}</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{status_line("Embedded Loaded",embedding_model)}</div>
            <div class="stat-label">{f"Using {embedding_model.model_name}" if embedding_model is not None else "Not Loaded"}</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{status_line("Re-ranker Loaded",reranker)}</div>
            <div class="stat-label">{f"Using {reranker.model}" if reranker is not None else "Not Loaded"}</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{status_line("Ready To Chat",True) if llm is not None and embedding_model is not None else ""}</div>
            <div class="stat-label">Load Model, Embedding Model and Re-ranker (optional) to start chatting</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


#  ==== Column Config ====
col_left, col_right = st.columns(2, gap="large")

# --- Left: LLM (model selector + generation params together) ---
with col_left:

    st.markdown("### Language Model")
    with st.expander("Model Selection & Parameters"):

        # — Model picker —
        st.markdown("##### Model")
        model_type = st.selectbox(
            "LLM",
            ["local", "Open-AI"],
            label_visibility="collapsed",
        )
        st.session_state.llm_config = build_llm_config(model_type)
        st.markdown("---")

        # — Generation params (live in the same expander) —
        st.markdown("##### Generation Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Higher = more creative; lower = more deterministic.",
        )
        max_tokens = st.slider(
            "Max Tokens",
            min_value=64, max_value=4096,
            value=st.session_state.max_tokens,
            step=64,
            help = "Higher = More tokens allowed in generation; lower = less tokens allowed for generations"
        )
        st.session_state.temperature = temperature
        st.session_state.max_tokens  = max_tokens

        st.markdown("---")

        # — Load button —
        if st.session_state.llm_config:
            if st.button("Load Model", key="load_llm"):
                with st.spinner("Loading…"):
                    st.session_state.llm = load_model(
                        st.session_state.llm_config,
                        st.session_state.max_tokens,
                        st.session_state.temperature,
                    )
                if st.session_state.llm:
                    st.success(f"Using {st.session_state.llm.model}")

    # — Embedding model —
    st.markdown("### Embedding Model")
    with st.expander("Embedding Models"):
        embedding_type = st.selectbox(
            "embedding_model",
            ["local","Open-AI"],
            label_visibility="collapsed",
        )

        st.session_state.embedding_config = build_embedding_config(embedding_type)
        if st.button("Load Embeddings", key="load_embed"):
            st.session_state.embedding_model = load_embedding_model(st.session_state.embedding_config)
            if st.session_state.embedding_model:
                st.success(f"Using `{st.session_state.embedding_model.model_name}`.")

# --- Right: Retrieval + Re-ranker ---
with col_right:

    st.markdown("### Retrieval")
    with st.expander("Retrieval Settings"):
        top_k = st.slider(
            "Top-K chunks returned",
            min_value=1, max_value=25,
            value=st.session_state.top_k,
            help="Number of chunks passed to the LLM as context.",
        )
        chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=64, max_value=1048,
            value=st.session_state.chunk_size,
            step=32,
        )
        chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0, max_value=216,
            value=st.session_state.chunk_overlap,
            step=8,
        )
        score_threshold = st.slider(
            "Score threshold",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="Minimum similarity score for a chunk to be included.",
        )
        st.session_state.top_k         = top_k
        st.session_state.chunk_size    = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        st.session_state.score_threshold = score_threshold

        chunk_method = st.selectbox(
            "Chunking Method",
            ["RecursiveCharacterTextSplitter","SentenceTransformerTokenTextSplitter"]
        )
        st.session_state.chunk_method = chunk_method
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size,chunk_overlap) if chunk_method == "RecursiveCharacterTextSplitter" else SentenceTransformersTokenTextSplitter(chunk_size) 
        st.markdown("### Re-ranker")
    with st.expander("Cohere Re-ranker  (optional)"):
        top_n = st.slider(
            "Top n",
            min_value=1, max_value=10,
            help="The n number of chunks to return",
        )
        st.markdown("##### Cohere API key")
        st.write("Using: rerank-english-v3.0")
        st.session_state.top_n = top_n
        st.session_state.cohere_api_key = st.text_input(
            "COHERE_API_KEY",
            type="password",
            value=st.session_state.cohere_api_key,
            label_visibility="collapsed",
            placeholder="COHERE_API_KEY",
        )
        if st.button("Load Re-ranker",key="load_reranker"):
            if st.session_state.cohere_api_key:
                st.session_state.reranker = load_reranker_model(st.session_state.cohere_api_key,top_n)
                if st.session_state.reranker:
                    st.success(f"Using: {st.session_state.reranker.model}")
                    pass
st.markdown('<hr class="divider">', unsafe_allow_html=True)


st.markdown("### Save Preferences")
with st.expander("Save Preferences"):

    save_name = st.text_input("Preference Name")

    if st.button("Save"):
        if not save_name:
            st.warning("Save name cannot be empty")
        else:
            created = st.session_state.preferences_db.add_preferences(
                uid = st.session_state.uid,
                save_name = save_name,
                temp = st.session_state.temperature,
                max_tokens = st.session_state.max_tokens,
                model = st.session_state.llm_config["model_name"],
                top_k = st.session_state.top_k,
                chunk_size = st.session_state.chunk_size,
                chunk_overlap = st.session_state.chunk_overlap,
                score_thresh = st.session_state.score_threshold,
                chunk_method = st.session_state.chunk_method,
                top_n = st.session_state.top_n
            )
            if created:
                st.success("Preferences saved")
            else:
                st.warning("Not saved")


st.markdown("### Load Preferences")
with st.expander("View Preferences"):

    preferences = st.session_state.preferences_db.load_preferences(
        uid=st.session_state.uid
    )

    if not preferences:
        st.markdown(
            '<div class="pref-empty">○ &nbsp; No saved preferences found.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="pref-grid">', unsafe_allow_html=True)

        for p in preferences:
            id, uid, save_name, temp, max_tokens, model, top_k, \
            chunk_size, chunk_overlap, score_thresh, chunk_method, top_n = p

            params = [
                ("Temperature",    temp),
                ("Max Tokens",     max_tokens),
                ("Top-K",          top_k),
                ("Chunk Size",     chunk_size),
                ("Chunk Overlap",  chunk_overlap),
                ("Score Thresh",   score_thresh),
                ("Chunk Method",   chunk_method),
                ("Top-N",          top_n),
            ]

            params_html = "".join(
                f"""
                <div class="pref-param">
                    <span class="pref-param-label">{label}</span>
                    <span class="pref-param-value">{value}</span>
                </div>
                """
                for label, value in params
            )

            st.markdown(
                f"""
                <div class="pref-card">
                    <div class="pref-card-header">
                        <span class="pref-card-name">{save_name}</span>
                        <span class="pref-card-model">{model}</span>
                    </div>
                    <div class="pref-params">
                        {params_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)


            # Load
            col_left, col_right = st.columns([2, 1], gap="large")
            with col_left:

                if st.button("Load", key=f"load_pref_{id}"):
                    st.session_state.temperature   = temp
                    st.session_state.max_tokens    = max_tokens
                    st.session_state.top_k         = top_k
                    st.session_state.chunk_size    = chunk_size
                    st.session_state.chunk_overlap = chunk_overlap
                    st.session_state.chunk_method  = chunk_method
                    st.session_state.top_n         = top_n
                    st.toast(f"Loaded '{save_name}'", icon="✦")

            # Delete 
            with col_right:
                if st.button("Delete", key = f"delete_pref_{id}"):
                    deleted = st.session_state.preferences_db.delete_preferences(
                        id = id
                    )
                    if deleted:
                        st.rerun()




            st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)






# ==== Data Uploading ====
st.markdown("### Upload Data Sources")
uploaded_files = st.file_uploader("Upload your PDF's",type = "pdf",accept_multiple_files=True)
if uploaded_files:
    with st.expander("Uploaded Status"):
        for file in uploaded_files:
            st.success(f"Uploaded {file.name}")
        

if st.session_state.uid:
    save_uploaded_files(uploaded_files,st.session_state.uid)
    

# ==== Create Persistent Directory ====
st.markdown("### Create Persistent Directory")

if st.button("Create Persistent Directory"):
    pass