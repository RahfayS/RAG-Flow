from db.user_db import UserDB
from db.model_preferences import ModelPreferencesDB

import streamlit as st
import uuid

def initialize_session_state():
    defaults = {
        "user": {},
        "llm": None,
        "embedding_model": None,
        "text_splitter": None,
        "reranker": None,
        "temperature": 0.2,
        "max_tokens": 512,
        "top_k": 5,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "score_threshold" : 0.2,
        "chunk_method": None,
        "text_splitter": None,
        "top_n": 3,
        "cohere_api_key": "",
        "openai_api_key": "",
        "chat_history": [],
        "user_db": UserDB(),
        "uid": None,
        "preferences_db": ModelPreferencesDB()
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Generate unique IDs ONLY if missing
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4()

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = uuid.uuid4()