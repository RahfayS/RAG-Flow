

from typing import TypedDict, Literal
import streamlit as st

class ModelConfig(TypedDict):
    type: Literal["local", "Open-AI"]
    model_name: str
    api_key:    str


def build_llm_config(model: str) -> ModelConfig | None:
    """Render model-specific widgets and return a config dict, or None."""
    match model:
        case "local":
            local_model = st.selectbox(
                "Available Models",
                ["qwen2.5:3b", "gemma2:2b",
                "llama3.2:3b", "other"],
                label_visibility="collapsed",
            )
            if local_model == "other":
                local_model = st.text_input("HuggingFace model ID")
            if local_model:
                return {"type": "local", "model_name": local_model}

        case "Open-AI":
            st.write("Model: gpt-4o-mini")
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password",
                                        value=st.session_state.openai_api_key,
                                        label_visibility="collapsed",
                                        placeholder="OPEN_API_KEY")
            if st.session_state.openai_api_key:
                return {"type": "Open-AI", "model_name": "gpt-4o-mini", "api_key": st.session_state.openai_api_key}
    return None


def build_embedding_config(embedding_model:str)->ModelConfig | None:
    
    match embedding_model:
        case "local":
            local_model = st.selectbox(
                "Available Model",
                ["jinaai/jina-embeddings-v4",
                "Qwen/Qwen3-Embedding-0.6B",
                "all-mpnet-base-v2"],
                label_visibility="collapsed",
            )
            return {"type":"local","model_name":local_model,"api_key":None}
        
        case "Open-AI":
            st.write("Model: text-embedding-3-large")
            st.session_state.openai_api_key = st.text_input("Embed OpenAI API key", type="password",
                                        value=st.session_state.openai_api_key,
                                        label_visibility="collapsed",
                                        placeholder="OPENAI_API_KEY")
            if st.session_state.openai_api_key:
                return {"type": "Open-AI", "model_name": "text-embedding-3-large", "api_key": st.session_state.openai_api_key}
    return None
