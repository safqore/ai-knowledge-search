from dotenv import load_dotenv
import os

# Load environment variables from .env file
# TODO: add instructions in readme.md to create a .env file with the LANGSMITH_API_KEY
load_dotenv()

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGCHAIN_PROJECT="ai-knowledge-search"

os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_TRACING"] = "true"

# Load PDF document via Langchain loaders - https://python.langchain.com/docs/how_to/document_loader_pdf/
from langchain_community.document_loaders import PyPDFLoader
str_pdf_path = "data/national-cdf-list-v1.331.pdf"
obj_loader = PyPDFLoader(str_pdf_path)
lst_documents = obj_loader.load()

# Extract all PDF content 
lst_pdf_content = [obj_document.page_content for obj_document in lst_documents]
str_pdf_content = " ".join(lst_pdf_content)
# print(str_pdf_content)

# Split PDF content into chunks for downsteam LLM processing - https://python.langchain.com/docs/concepts/text_splitters/
# See for chunking visualisation: https://chunkviz.up.railway.app/
from langchain_text_splitters import RecursiveCharacterTextSplitter
obj_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
lst_chunks = obj_text_splitter.split_text(str_pdf_content)
# print(lst_chunks[0])

# Create embeddings for the text chunks
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain.docstore import InMemoryDocstore

dimension = 768  # This should match the dimension of your embeddings
index = IndexFlatL2(dimension)
docstore = InMemoryDocstore({})
index_to_docstore_id = {}
vector_store = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
text_ids = vector_store.add_texts(lst_chunks)

print(text_ids[:3])

# from langchain.vectorstores import FAISS
# vector_store = FAISS.from_texts(lst_chunks, embeddings)

# # Search for a keyword or phrase
# query = "Alectinib"
# results = vector_store.similarity_search(query)

# # Print the search results
# for result in results:
#     print(result.page_content)


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGING_FACE_HUB_TOKEN')

# obj_llm = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# chat_model = ChatHuggingFace(llm=obj_llm)

# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
# )

# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(
#         content="What happens when an unstoppable force meets an immovable object?"
#     ),
# ]

# ai_msg = chat_model.invoke(messages)

# print(ai_msg.content)

# ############################

# Summarize the text chunks
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# question_prompt = PromptTemplate.from_template(
#     'What is criteria for use for {drug}?'
# )
# question_chain =  question_prompt | obj_llm

# summaries = [question_chain.run(text=chunk) for chunk in lst_chunks]

# # Print the summaries
# for summary in summaries:
#     print(summary)