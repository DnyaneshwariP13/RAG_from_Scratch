from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
import os

def process_documents(embeddings):
    """Process documents and create vector store"""
    try:
        loader = TextLoader("transcript.txt")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=40
        )
        split_docs = text_splitter.split_documents(documents)
        
        return DocArrayInMemorySearch.from_documents(
            split_docs, 
            embeddings
        )
    except Exception as e:
        raise RuntimeError(f"Document processing failed: {str(e)}")

def get_embeddings():
    """Initialize embedding model"""
    try:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="models/embedding-001"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding initialization failed: {str(e)}")