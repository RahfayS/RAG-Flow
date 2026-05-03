
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank, CohereEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.schema import Document
from langchain_chroma import Chroma

from utils.load_model import load_model
from config import LOGS_DIR,PATH_TO_CHAT_MEMORY,VECTOR_DB_DIR,DATA_DIR, ROOT

from typing import TypedDict, Annotated
from dotenv import load_dotenv
import streamlit as st
import requests
import logging
import time
import json
import os

# uv run main.py hs-rag

# --- Define Logging Stuff ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
log_file = os.path.join(LOGS_DIR, "app.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
    force=True
)
logger = logging.getLogger(__name__)

class InterpretableStructure(TypedDict):
    """Structured output for interpretable RAG responses"""
    answer: Annotated[str,"Direct and concise answer to the user's question based ONLY on retrieved context"]
    reasoning: Annotated[str,"Step-by-step explanation of how the answer was derived from the sources" ]

class QueryOptimizationStructure(TypedDict):
    """Structured output for query expansion"""
    queries: Annotated[list[str], "List of all expanded queries based on the users question"]

class RAGManager():
    def __init__(self,temp_paths:list[str],embedding_model:str = "embed-v4.0",reranker_model:str = "rerank-english-v3.0",chunk_size:int = 500, chunk_overlap:int = 100,top_n:int=5):

        self.temp_paths = temp_paths
        # --- Load Environment variables ---
        load_dotenv()
        api_key = os.getenv("COHERE_API")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set")

        # --- Load LLM ---
        self.llm = load_model(max_tokens=216)
        self.llm_structured = self.llm.with_structured_output(InterpretableStructure)
        
        self.query_expanded_llm = load_model(max_tokens=516).with_structured_output(QueryOptimizationStructure)
        
        # --- Define the Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

        # --- Define BM25 ---
        self.bm25_retriever = None

        # --- Define Embedding Model ---
        
        self.embedding_model = CohereEmbeddings(
            model = embedding_model,
            cohere_api_key=api_key
        )

        # --- Define Reranker ---
        self.reranker = CohereRerank(
            model=reranker_model,
            top_n=top_n,
            cohere_api_key=api_key
        )        
        # --- Persistent Directory ---
        self.persistent_directory = os.path.join(VECTOR_DB_DIR,"chromadb")

       # --- Define data source path ---
        self.data_source_path = os.path.join(DATA_DIR,"documents")
        os.makedirs(self.data_source_path,exist_ok=True)

        #self.save_path = os.path.join(self.data_source_path,f"{self.source_name}.pdf")

        # --- Define Templates ---

        # --- Query Expansion Prompt ---
        self.query_expansion_prompt = PromptTemplate(
            template="""
            You are an expert at reformulating user questions about the book *Percy Jackson: The Sea of Monsters*.

            Your task is to perform query expansion to improve document retrieval in a RAG system.

            Given a user question, generate multiple alternative search queries that:
            - Preserve the original meaning
            - Use different wording, phrasing, and synonyms
            - Include relevant character names, locations, or events when helpful
            - Vary specificity (broad -> narrow)

            Instructions:
            - Generate {n} diverse query variations
            - Do NOT answer the question
            - Do NOT add new facts not implied by the original question
            - Keep each query concise and focused
            - Avoid redundancy

            User Question:
            {question}
        """,
        input_variables=["question","k"]
        )


        # --- System Prompt ---
        self.system_prompt = """
        You are a knowledgeable literary assistant answering questions about the book *Percy Jackson - The Sea of Monsters* using the provided context.

        Context:
        {context}

        User Question:
        {question}

        Instructions:
        - Answer ONLY using the provided context. Do not use outside knowledge.
        - If the context does not contain enough information, explicitly say so.
        - When relevant, extract and clearly present:
        - Characters involved
        - Key events or actions
        - Important dialogue or quotes
        - Settings or locations
        - Motivations or relationships between characters
        - If multiple passages or sources are relevant, synthesize them into a clear and coherent answer.
        - Focus on reasoning and explanation, not just copying text.
        - Use concise but informative explanations.
        - If helpful, structure the answer using bullet points or short paragraphs.
        - Do NOT hallucinate missing details.
        - Focus only on the root cause of the problem
        - Ignore background events unless they directly answer the question
        - Prefer explanations over descriptions of attacks or events

        Output:
        Provide a clear, well-structured answer grounded in the context.
        """
    
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",self.system_prompt),
            ("human","Context: {context}, {question}")
        ])


        # --- Define Save path for answers ---
        self.answer_path = os.path.join(DATA_DIR,"rag_flow")
        if not os.path.exists(self.answer_path):
            os.makedirs(self.answer_path)

        self.db = None

    def load_documents(self) -> list[Document]:
        """Loads PDF data source"""
        all_docs = []
        for tp in self.temp_paths:
            logger.info(tp)
            path = os.path.join(ROOT,tp)
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata['source_document'] = path
            all_docs.extend(loader.load())
        return all_docs
        

    def __batch_docs(self,docs,size:int = 20):
        """Batches docs in size of 20 to help against rate limiters"""
        for i in range(0, len(docs), size):
            yield docs[i:i+size] 

    def build_vector_db(self):
        """Builds a vector database from the data source"""
        # --- Verify Persistent directory hasnt been built ---
        if os.path.exists(self.persistent_directory) and len(os.listdir(self.persistent_directory)) != 0:
            logger.info(f'[INFO]: Persistent Directory already exists')
            return
        # --- Create Persistent Directory ---
        os.makedirs(self.persistent_directory,exist_ok=True)
        # --- Load Documents ---
        documents = self.load_documents()

        # --- Split docs into chunks ---
        chunks = self.text_splitter.split_documents(documents)

        # --- Store Chunks ---
        for batch in self.__batch_docs(chunks):
            if self.db is None:
                self.db = Chroma.from_documents(
                    batch,self.embedding_model, persist_directory=self.persistent_directory
                )
            else:
                self.db.add_documents(batch)
            time.sleep(1)

        logger.info(f"Chroma DB Created {self.db._collection.count()}")

    def __build_context(self,retrieved_docs):
        """Add context to the top k retrieved docs"""
        context = ""
        sources = []

        # --- Iterate through all chunks in the retrieved docs ----
        for i,chunk in enumerate(retrieved_docs,1):

            # --- Extract Meta data ---
            page = chunk.metadata.get("page","n/a") # get page number 
            source = chunk.metadata.get("source_document","PDF")

            # --- Define the context that we will pass to the llm ---
            context += f"\nChunk {i}: {chunk.page_content}"

            # --- Structure the source meta data ---
            sources.append({
                "chunk":i,
                "page": page,
                "source":source,
                "page_content":chunk.page_content
            })

        return context,sources
    
    def __expand_query(self,q:str,n:int = 5) -> list[str]:
        """Expands query to n number of permutations"""
        chain = self.query_expansion_prompt | self.query_expanded_llm
        return chain.invoke({"n":n,"question":q})

    @st.cache_resource # Loads vector store once, and caches it across sessions
    def load_vector_stores(self):
        """Loads chroma vector store"""
        # ---- Ensure directory exists and is not empty ----
        if not os.path.exists(self.persistent_directory) or not os.listdir(self.persistent_directory):
            logger.info("Persistent directory missing or empty, building DB")
            self.build_vector_db()
        else:
            logger.info("Persistent directory exists")
            return
        # --- Load DB ---
        self.db = Chroma(
                persist_directory=self.persistent_directory,
                embedding_function=self.embedding_model
            )
        count = self.db._collection.count()
        logger.info(f"Chroma DB Loaded {count}")

        # --- Handle Empty DB ---
        if count == 0:
            logger.warning("Chroma DB is empty, rebuilding")
            self.build_vector_db()
            self.db = Chroma(
                persist_directory = self.persistent_directory,
                embedding_function = self.embedding_model
            )
        # ---- Retrieve all docs ---
        all_docs = self.db.get(include=["documents","metadatas"])
        docs = [ # Convert to Document objects
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        logger.info(f'All docs: {len(docs)}')
        # --- BM25 Docs ---
        # ---- BM25 ----
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(documents=docs)
        else:
            logger.warning("No docs available, BM25 disabled")
            self.bm25_retriever = None

        

    def process_requests(self,k:int = 25) -> None:
        """Runs user requests on vector store"""
        if not os.path.exists(self.persistent_directory):
            self.build_vector_db()
        else:
            self.load_vector_stores()
        file_path = os.path.join(self.answer_path, "sea_of_monsters.json")


        # --- Load existing data (if file exists) ---
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = []

        # --- Define Vector Retriever ---
        vector_retriever = self.db.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":k}
        )

        # -- Hybrid Retriever ---
        hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever,self.bm25_retriever],
                weights=[0.5,0.5] # Weights controlling how much each retriever influences rankings
        )


        # --- Run Generation ---
        print('\n===== Interpretable RAG (Percy Jackson The Sea of Monsters) =====')
        while True:
            query = input("\nEnter a query or press q to quit: ").lower().strip()
            if query == "q":
                return

            # --- Enhance Query ---
            expanded_queries = self.__expand_query(query)
            logger.info(f"Enhanced queries: {expanded_queries['queries']}")


            print("\n\n===== Enhanced Queries =====\n")
            print(f'Original Query: {query}')
            print(" Enhanced Queries:\n" + "\n".join(e for e in expanded_queries['queries']))

            # --- Retrieve relevant docs for each enhanced query ---
            retrieved_docs = []
            for e in expanded_queries["queries"]:
                retrieved_docs.extend(hybrid_retriever.invoke(e))

            # --- Remove Duplicate Chunks ---
            seen = set()
            unique_docs = []
            for doc in retrieved_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            logger.info(f'Number of unique docs: {len(unique_docs)}')

            # --- Re-rank the retrieved docs ---
            reranked_docs = self.reranker.compress_documents(unique_docs,query)
            print(f'\n\n===== Re-ranked Docs ======\n')
            print("Re-ranked Docs:\n" + "\n".join(r.page_content for r in reranked_docs))
            
            # --- Build Context ---
            context,sources = self.__build_context(reranked_docs)

            # --- Create Prompt ---
            prompt = self.prompt_template.invoke({
                "context":context,
                "question":query
            })

            # --- Get response ---
            response = self.llm_structured.invoke(prompt)

            # --- Add Question to Response ---
            response['question'] = query

            # --- Log Sources ---
            for i,src in enumerate(sources,1):
                logger.info(f'\n------------------------------------------\n')
                logger.info(f"Chunk: {src['chunk']} \t Page: {src['page']}")
                logger.info(f'Content: {src['page_content']}')
                logger.info(f'\n------------------------------------------\n')

                # --- Add meta data to JSON response ---
                response[f'chunk_{i}'] = f"Page {src['page']}: {src['page_content']}"

            # --- Add new response ---
            data.append(response)

            # --- Save the response ---
            with open(file_path,'w') as f:
                json.dump(data,f,indent=4)

            # --- Display Info ---
            print("\n===== Model Response =====\n")
            print(f'Response: {response["answer"]}')
            print(f'Reasoning: {response["reasoning"]}')