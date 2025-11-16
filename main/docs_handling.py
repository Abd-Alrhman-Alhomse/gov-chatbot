import os
from typing import Dict, List, Optional, Annotated, TypedDict
from dataclasses import dataclass
import json
import logging
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
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocState(TypedDict):
    """State for document ingestion flow"""
    document_content: str
    chunks: List[Document]
    embeddings_created: bool


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


class docHandling:
    def __init__(self):
        # Initialize Gemini models
        print('hello')
        print(settings.EMBEDDING_MODEL)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)

        print('hello 1')
        # initialize chromaDB

        # try:
        #     print("hello")
        #     print(settings.EMBEDDING_MODEL)
        #     self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        #     print("hello 1")
        # except Exception as e:
        #     import traceback
        #     traceback.print_exc()
        try:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
            self.collection_name = settings.COLLECTION_NAME
            self.chroma_ef_instance = GeminiEmbeddingFunction(self.embeddings)


            self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.chroma_ef_instance
                )
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            print('first')
        except:
            logger.error(f"Loaded ChromaDB collection: {self.collection_name} fail")
            print('second')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )


        print('done')

        # Initialize graphs
        self.doc_graph = self._build_document_flow()



    def _build_document_flow(self) -> StateGraph:
        """Build the document ingestion flow"""

        workflow = StateGraph(DocState)

        workflow.add_node("process_document", self._process_document)
        workflow.add_node("create_chunks", self._create_chunks)
        workflow.add_node("create_embeddings", self._create_embeddings)

        workflow.add_edge(START, "process_document")
        workflow.add_edge("process_document", "create_chunks")
        workflow.add_edge("create_chunks", "create_embeddings")
        workflow.add_edge("create_embeddings", END)

        return workflow.compile()


    # Document Flow Methods
    def _process_document(self, state: DocState) -> DocState:
        """Initial document processing"""
        logger.info(f"Start Processing document")

        return state


    def _create_chunks(self, state: DocState) -> DocState:
        """Create document chunks for embedding"""
    
            # Otherwise, use the original document content
        content = state['document_content']
        logger.info("Using original document content for chunking.")
        print(content)
        chunks = self.text_splitter.split_text(content)
        state['chunks'] = [
            Document(
                page_content=chunk,
                metadata={
                    "timestamp": datetime.now().isoformat()
                }
            )
            for chunk in chunks
        ]

        logger.info(f"Created {len(state['chunks'])} chunks")

        return state

    def _create_embeddings(self, state: DocState) -> DocState:
        """Create and store embeddings"""
        if not state['chunks']:
            logger.warning("No chunks to embed and add to ChromaDB.")
            state['embeddings_created'] = False
            return state

        # Prepare data for ChromaDB
        # Generate unique IDs for the documents being added to ChromaDB
        ids = [f"{i}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" for i in range(len(state['chunks']))]
        documents = [chunk.page_content for chunk in state['chunks']]
        metadatas = [chunk.metadata for chunk in state['chunks']]

        # Add documents to ChromaDB collection
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
                # ChromaDB will use the embedding_function provided during collection creation
            )
            state['embeddings_created'] = True
            logger.info(f"Added {len(documents)} documents to ChromaDB collection.")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            state['embeddings_created'] = False
        return state

    # Public Methods
    def add_document(self, content: str) -> Dict:
        """Add a document to the knowledge base"""
        initial_state = DocState(
            document_content=content,
            chunks=[],
            embeddings_created=False,
        )

        result = self.doc_graph.invoke(initial_state)

        return {
            "success": result['embeddings_created'],
            "chunks_created": len(result['chunks']),
        }