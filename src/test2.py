from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read PDF
pdfreader = PdfReader("data/raw/national-cdf-list-v1.331.pdf")
raw_text = ' '.join(page.extract_text() for page in pdfreader.pages if page.extract_text())

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
vectorstore = FAISS.from_texts(texts, embeddings)

# Load model and create pipeline
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

def answer_question(question, context):
    return qa_pipeline(question=question, context=context, max_length=512, truncation=True)

# Example usage
query = "What are the indications of Alectinib?"
relevant_docs = vectorstore.similarity_search(query, k=2)
context = " ".join(doc.page_content for doc in relevant_docs)

result = answer_question(query, context)

print(f"Question: {query}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2f}")