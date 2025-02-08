import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGCHAIN_PROJECT="ai-knowledge-search"

os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_TRACING"] = "true"

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))

# Search for a keyword or phrase
query = "What is criteria for use for Alectinib?"
query_vector = embeddings.embed_query(query)

results = qdrant_client.search(
    collection_name="cancer_drugs_fund_list",
    query_vector=query_vector,
    limit=10,
    with_payload=True,
)

# Print the search results
for result in results:
    print(result.payload["text"])

# Set up HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGING_FACE_HUB_TOKEN')

# Initialize HuggingFace endpoint
obj_llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=obj_llm)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)

print(ai_msg.content)

# # Summarize the text chunks
# question_prompt = PromptTemplate.from_template(
#     'What is criteria for use for {drug}?'
# )
# question_chain = question_prompt | obj_llm

# summaries = [question_chain.run(text=chunk) for chunk in lst_chunks]

# # Print the summaries
# for summary in summaries:
#     print(summary)