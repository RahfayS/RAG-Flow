from langchain_cohere import CohereRerank, CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import BaseDocumentCompressor,Document
from utils.build_config import ModelConfig

from openai import OpenAI
import streamlit as st
import subprocess


def __model_exists(model_name:str) -> bool:
    """Verfies a local model already exists on device"""
    result = subprocess.run(
        ["ollama","list"],
        capture_output=True,
        text=True
    )

    models = [
        line.split()[0] 
        for line in result.stdout.splitlines()[1:]  # skip header
    ]

    return model_name in models



def validate_openai_key(api_key: str) -> tuple[bool, str]:
    try:
        client = OpenAI(api_key=api_key)

        # lightweight request
        client.models.list()

        return True, "Valid API key"

    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_model(llm_config:ModelConfig,max_tokens:int=128,temperature:float = 0.2,default_model:str = "qwen2.5:3b") -> BaseChatModel | None :
    """Loads LLM"""
    model_type = llm_config["type"]
    st.write(model_type)
    try:
        match model_type:
            case "local":
                model_name = llm_config["model_name"]
                if not __model_exists(model_name):
                    st.write(f'{model_name} does not exist, pulling...')
                    subprocess.run(["ollama", "pull", model_name], check=True)
                
                return ChatOllama(
                    model = model_name,
                    num_predict=max_tokens,
                    temperature=temperature
                )
        
            case "Open-AI":
                valid,msg = validate_openai_key(llm_config["api_key"])
                if valid:
                    return ChatOpenAI(
                        name = llm_config["model_name"],
                        api_key=llm_config["api_key"],
                        max_tokens = max_tokens,
                        temperature=temperature
                    )
                else:
                    st.error(f'Error {msg}')
                    st.write(f'Using Default Local Model')
                    return ChatOllama(
                    model = default_model,
                    num_predict=max_tokens,
                    temperature=temperature
                )
    
    except Exception as e:
        st.error(f'Error {e}, using default')
        return ChatOllama(
                    model = default_model,
                    num_predict=max_tokens,
                    temperature=temperature
                )
    
@st.cache_resource
def load_embedding_model(embedding_config:ModelConfig,default_embedding:str = "Qwen/Qwen3-Embedding-0.6B"):
    """Loads Embedding Model"""
    if embedding_config is None:
        st.write("Define Configuration first")
        return
    model_type = embedding_config["type"]
    try:
        match model_type:

            case "local":
                return HuggingFaceEmbeddings(
                    model_name = embedding_config["model_name"]
                )
    
            case "Open-AI":
                valid,msg = validate_openai_key(embedding_config["api_key"])
                if valid:
                    return OpenAIEmbeddings(
                        model = embedding_config['model_name']
                    )
                else:
                    st.error(f'Error {msg}')
                    st.write(f'Using Default Local Model')
                    return HuggingFaceEmbeddings(
                    model_name = default_embedding
                )
            
    except Exception as e:
        st.error(f'Error {e}, using default')
        return HuggingFaceEmbeddings(
            model_name = default_embedding
        )
        


def __validate_reranker(reranker:BaseDocumentCompressor)->bool:
    """Validates reranker model is loaded"""
    try:
        test_doc = [Document(
            page_content="Hello World"
        )]
        query = "What is the message"
        reranked = reranker.compress_documents(documents=test_doc,query=query)
        if reranked:
            return True
        return False
    except Exception as e:
        if hasattr(e,'body'):
            st.error(e.body["message"])
        elif hasattr(e,dict):
            st.error(e['body']['message'])
        else:
            st.error(f'Error {e}')
        return False

@st.cache_resource
def load_reranker_model(api_key: str,top_n: int,reranker_model: str = "rerank-english-v3.0")-> BaseDocumentCompressor:
    try:
        reranker = CohereRerank(
            model=reranker_model,
            top_n=top_n,
            cohere_api_key=api_key
        )
        if __validate_reranker(reranker):
            return reranker   
        return None

    except Exception as e:
        st.error(f"Error loading reranker: {e}")
        return None