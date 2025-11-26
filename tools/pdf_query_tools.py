from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import tool
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os


def get_llm_model():
    """
    Get the Google Gemini LLM model
    """
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@tool
def indian_constitution_pdf_query(query: str) -> str:
    """Returns a related answer from the Indian Constitution PDF using semantic search from input query"""
    
    llm = get_llm_model()
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_constitution",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:
        # Create new vector database if not exists
        reader = PdfReader("tools/data/constitution.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings_model)
        
        # Create directory if it doesn't exist
        os.makedirs("db", exist_ok=True)
        db.save_local("db/faiss_index_constitution")

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)

    return result


@tool
def indian_laws_pdf_query(query: str) -> str:
    """Returns a related answer from the "THE BHARATIYA NYAYA (SECOND) SANHITA, 2023" PDF which states all of the laws of India, using semantic search from input query"""
    
    llm = get_llm_model()
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_bns",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:
        # Create new vector database if not exists
        reader = PdfReader("tools/data/BNS.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings_model)
        
        # Create directory if it doesn't exist
        os.makedirs("db", exist_ok=True)
        db.save_local("db/faiss_index_bns")

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)

    return result


# Enhanced versions with QA chain support (optional)
@tool
def indian_constitution_pdf_query_with_qa(query: str) -> str:
    """Returns a processed answer from the Indian Constitution PDF using semantic search and QA chain"""
    
    llm = get_llm_model()
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_constitution",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:
        # Create new vector database if not exists
        reader = PdfReader("tools/data/constitution.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings_model)
        os.makedirs("db", exist_ok=True)
        db.save_local("db/faiss_index_constitution")

    # Get relevant documents
    docs = db.similarity_search(query, k=4)
    
    # Use QA chain for better answers
    try:
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        result = qa_chain.run(input_documents=docs, question=query)
        return result
    except Exception as e:
        # Fallback to simple retrieval if QA chain fails
        return str(docs)


@tool
def indian_laws_pdf_query_with_qa(query: str) -> str:
    """Returns a processed answer from the BNS PDF using semantic search and QA chain"""
    
    llm = get_llm_model()
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    try:
        db = FAISS.load_local("db/faiss_index_bns",
                              embeddings_model, allow_dangerous_deserialization=True)
    except:
        # Create new vector database if not exists
        reader = PdfReader("tools/data/BNS.pdf")
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings_model)
        os.makedirs("db", exist_ok=True)
        db.save_local("db/faiss_index_bns")

    # Get relevant documents
    docs = db.similarity_search(query, k=4)
    
    # Use QA chain for better answers
    try:
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        result = qa_chain.run(input_documents=docs, question=query)
        return result
    except Exception as e:
        # Fallback to simple retrieval if QA chain fails
        return str(docs)