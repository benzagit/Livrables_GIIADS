"""Load Chroma retriever for RAG."""
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DIR = Path(__file__).resolve().parent.parent / "vector_db"


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever, vectorstore
