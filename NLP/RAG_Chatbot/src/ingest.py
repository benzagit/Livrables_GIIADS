"""Ingest rock-paper-scissors docs into a local Chroma vector store."""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTOR_DIR = Path(__file__).resolve().parent.parent / "vector_db"


def load_documents():
    exts = [".txt", ".md"]
    docs = []
    for path in DATA_DIR.glob("*"):
        if path.suffix.lower() in exts:
            docs.append(TextLoader(str(path), encoding="utf-8").load())
    # flatten list of lists
    return [item for sublist in docs for item in sublist]


def build_vector_store():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    print(f"Saved {len(splits)} chunks to {VECTOR_DIR}")


if __name__ == "__main__":
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    build_vector_store()
