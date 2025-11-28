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
import threading


_embed_lock = threading.Lock()
_embeddings_model = None
_db_constitution = None
_db_bns = None

def _get_embeddings():
    """Singleton embeddings to avoid re-instantiation per tool call."""
    global _embeddings_model
    if _embeddings_model is None:
        with _embed_lock:
            if _embeddings_model is None:
                _embeddings_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
    return _embeddings_model

def _load_or_build_faiss(index_dir: str, pdf_path: str):
    """Load FAISS index from disk, or build once and persist.

    Returns a FAISS vectorstore.
    """
    embeddings_model = _get_embeddings()
    try:
        return FAISS.load_local(index_dir, embeddings_model, allow_dangerous_deserialization=True)
    except Exception:
        reader = PdfReader(pdf_path)
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings_model)
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)
        db.save_local(index_dir)
        return db


@tool
def indian_constitution_pdf_query(query: str) -> str:
    """Returns a related answer from the Indian Constitution PDF using semantic search from input query"""
    
    global _db_constitution
    if _db_constitution is None:
        _db_constitution = _load_or_build_faiss("db/faiss_index_constitution", "tools/data/constitution.pdf")

    retriever = _db_constitution.as_retriever(search_kwargs={"k": 3})
    result = retriever.invoke(query)

    return result


@tool
def indian_laws_pdf_query(query: str) -> str:
    """Returns a related answer from the "THE BHARATIYA NYAYA (SECOND) SANHITA, 2023" PDF which states all of the laws of India, using semantic search from input query"""
    
    global _db_bns
    if _db_bns is None:
        _db_bns = _load_or_build_faiss("db/faiss_index_bns", "tools/data/BNS.pdf")

    retriever = _db_bns.as_retriever(search_kwargs={"k": 3})
    result = retriever.invoke(query)

    return result


# Enhanced versions with QA chain support (optional)
@tool
def indian_constitution_pdf_query_with_qa(query: str) -> str:
    """Returns a processed answer from the Indian Constitution PDF using semantic search and QA chain"""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings_model = _get_embeddings()

    db = _load_or_build_faiss("db/faiss_index_constitution", "tools/data/constitution.pdf")

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
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings_model = _get_embeddings()

    db = _load_or_build_faiss("db/faiss_index_bns", "tools/data/BNS.pdf")

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