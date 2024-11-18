# TODO: Move to correct location in the project structure or remove if not needed

import os
import pdfplumber
import pandas as pd
import pinecone
import torch
import time
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Set up Pinecone client
pc_client = PineconeClient(os.environ.get("PINECONE_API_KEY"))  # Replace with your actual API key

# Define index name and embedding dimension
index_name = "nhs-cdf-index"
dimension = 384  # Appropriate for the "all-MiniLM-L6-v2" model

# Check if the Pinecone index exists; create if it does not
try:
    if index_name not in pc_client.list_indexes():
        pc_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")
except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
    if e.status == 409 and "ALREADY_EXISTS" in e.body:
        print(f"Index '{index_name}' already exists. Skipping creation.")
    else:
        raise  # Re-raise the exception if it's not an "already exists" conflict

# Hugging Face model for embeddings and QA pipeline
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
qa_model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name, device=device, clean_up_tokenization_spaces=False)

# Extract and embed text
def extract_text_from_pdf(pdf_path, start_page, end_page):
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num]
            text = page.extract_text()
            if text:
                text_content.append(text)
    return " ".join(text_content)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Upsert vectors in chunks to avoid size limits
def upsert_with_retry(index, vectors, chunk_size=100, retries=5, delay=2):
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i + chunk_size]
        for attempt in range(retries):
            try:
                index.upsert(chunk)
                break  # If successful, move to the next chunk
            except pinecone.core.openapi.shared.exceptions.ServiceException as e:
                if e.status == 503:  # Service Unavailable
                    print(f"Service Unavailable. Retrying in {delay} seconds... (Attempt {attempt + 1} of {retries})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise  # Re-raise if it's a different error
            except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
                if e.status == 400 and "message length too large" in e.body:
                    print("Batch size is too large, consider reducing the chunk_size.")
                    return
                else:
                    raise

# Text processing and embedding for Pinecone storage
pdf_path = "data/raw/national-cdf-list-v1.331.pdf"  # Updated path to the PDF file
document_text = extract_text_from_pdf(pdf_path, start_page=2, end_page=3)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(document_text)

# Generate embeddings and prepare for upsert
index = pc_client.Index(index_name)
vectors = [(f"doc_{i}", embed_text(chunk)) for i, chunk in enumerate(text_chunks)]
upsert_with_retry(index, vectors, chunk_size=50)  # Adjust chunk_size to stay under 4 MB limit

# Add this class
class BertEmbeddings:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
# Create an instance of BertEmbeddings
bert_embeddings = BertEmbeddings(embedding_model, tokenizer)

# Update vector store initialization
vector_store = PineconeVectorStore(
    index_name=index_name, 
    embedding=bert_embeddings,
)

# Define a custom retrieval chain with Hugging Face QA model
class HuggingFaceRetrievalQA:
    def __init__(self, retriever, qa_pipeline):
        self.retriever = retriever
        self.qa_pipeline = qa_pipeline

    def run(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        
        if context:
            result = self.qa_pipeline(question=query, context=context)
            return result['answer']
        return "No relevant information found."

# Initialize retrieval and QA chain
retriever = vector_store.as_retriever()
qa_chain = HuggingFaceRetrievalQA(retriever=retriever, qa_pipeline=qa_pipeline)

llm = HuggingFacePipeline(pipeline=qa_pipeline)
retriever = vector_store.as_retriever()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Ask a question
query = "What is the recommended drug for a specific condition?"
# Update query
response = qa_chain.invoke({"query": query})
print(response['result'])