import os
from typing import Dict, List, Optional, Annotated, TypedDict
from dataclasses import dataclass
import json
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
# import sys
# import os
# import django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot.settings')
# django.setup()
from django.conf import settings
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model

    def __call__(self, input_texts: Documents) -> Embeddings:
        # Ensure input_texts is a list of strings
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # ChromaDB might pass a single string or a list depending on the operation
        return self.embeddings_model.embed_documents(input_texts)

    # Add a name method as required by ChromaDB's protocol
    def name(self) -> str:
        return "google-gemini-embedding"

class ChatState(TypedDict):
    """State for the chat flow"""
    question: str
    retrieved_docs: List[Document]
    answer: str

class chatBot:
    def __init__(self):
        # Initialize Gemini models
        self.llm = GoogleGenerativeAI(model=settings.LLM_MODEL , temperature=0.1)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        # initialize chromaDB
        try:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
            self.collection_name = settings.COLLECTION_NAME
            self.chroma_ef_instance = GeminiEmbeddingFunction(self.embeddings)


            self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.chroma_ef_instance
                )
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except:
            logger.error(f"Loaded ChromaDB collection: {self.collection_name} fail")
        # Initialize graph
        self.chat_graph = self._build_chat_flow()


    def _build_chat_flow(self) -> StateGraph:
        """Build the chat interaction flow"""
        workflow = StateGraph(ChatState)

        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer")

        workflow.add_edge("generate_answer", END)

        return workflow.compile()


    def _retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents from vector database"""
        # Check if the collection exists and has documents
        try:
            # Check collection count directly
            count = self.collection.count()
            if count == 0:
                logger.warning("ChromaDB collection is empty. Cannot retrieve documents.")
                state['retrieved_docs'] = []
                return state
        except Exception as e:
            logger.error(f"Error accessing ChromaDB collection count: {e}")
            state['retrieved_docs'] = []
            return state

        # Create search query
        search_query = state['question']

        # Prepare filter based on classified module
        # query_filter = {"module": state['classified_module']} if state['classified_module'] != "General" else {}

        try:
            # Retrieve documents using ChromaDB's query method
            # Note: ChromaDB's query method returns dictionaries, not Document objects
            results = self.collection.query(
                query_texts=[search_query],
                n_results=5,
                # where=query_filter
            )

            # Convert ChromaDB results to Document objects
            retrieved_docs = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    retrieved_docs.append(
                        Document(
                            page_content=results['documents'][0][i],
                            metadata=results['metadatas'][0][i]
                        )
                    )

            state['retrieved_docs'] = retrieved_docs
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents from ChromaDB")

        except Exception as e:
            logger.error(f"Error retrieving documents from ChromaDB: {e}")
            state['retrieved_docs'] = []

        return state

    def _generate_answer(self, state: ChatState) -> ChatState:
        """Generate answer based on retrieved documents"""
        if not state['retrieved_docs']:
            # If no specific docs found, provide general system info
            context = ""
        else:
            
            context = "\n\n".join([doc.page_content for doc in state['retrieved_docs']])

        prompt = f"""

        the Context:
        {context}

        User Question: {state['question']}


        Answer:
        """

        answer = self.llm.invoke(prompt)
        state['answer'] = answer.content if hasattr(answer, 'content') else str(answer)


        return state





    def chat(self, question: str, session_id: str = "default") -> Dict:
        """Process a chat message"""
        initial_state = ChatState(
            question=question,
            retrieved_docs=[],
            answer="",
        )

        config = {"configurable": {"thread_id": session_id}}
        result = self.chat_graph.invoke(initial_state, config)

        return {
            "answer": result['answer'],
            "sources": len(result['retrieved_docs']),
        }