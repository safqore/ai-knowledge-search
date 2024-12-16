import logging
import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

def load_pdf(str_pdf_path: str) -> List:
    return PyPDFLoader(str_pdf_path).load()

def chunk_pdf(str_pdf_content: str) -> List[str]:
    obj_text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=76)
    return obj_text_splitter.split_text(str_pdf_content)

def setup_vectors_collection(db_client: QdrantClient, collection_name: str, dimension: int) -> None:
    if not db_client.collection_exists(collection_name):
        db_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

def update_vectors_db(lst_chunks: List[str]) -> bool:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        dimension = 768
        collection_name = "cancer_drugs_fund_list"

        qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
        setup_vectors_collection(qdrant_client, collection_name, dimension)
        
        vectors = [embeddings.embed_query(chunk) for chunk in lst_chunks]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {"id": i, "vector": vector, "payload": {"text": lst_chunks[i]}}
                for i, vector in enumerate(vectors)
            ],
        )
        return True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

def main():
    logging.info("Loading National Cancer Drugs Fund List PDF ...")
    lst_documents = load_pdf("data/national-cdf-list-v1.331.pdf")

    lst_pdf_content = [obj_document.page_content for obj_document in lst_documents]
    str_pdf_content = " ".join(lst_pdf_content)

    logging.info("Chunking PDF content for vector update / creation ...")
    lst_chunks = chunk_pdf(str_pdf_content)

    logging.info("Updating vectors database ...")
    status = update_vectors_db(lst_chunks)
    if status:
        logging.info("Vectors database updated successfully.")
    else:
        logging.error("Failed to update vectors database.")

if __name__ == "__main__":
    main()