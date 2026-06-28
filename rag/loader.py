"""PDF loading and text chunking.

Uses LangChain's PyPDFLoader for extraction and
RecursiveCharacterTextSplitter for intelligent chunking.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_chunk_pdf(file_path: str):
    """Load a PDF and split it into overlapping text chunks.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects, each containing a chunk
        with metadata (source filename, page number).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)
