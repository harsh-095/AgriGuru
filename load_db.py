import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load CSV as documents
loader = CSVLoader(file_path='resources\Crop_recommendation.csv')
# E:\InfyHackathon\Crop_recommendation.csv
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.split_documents(docs)


print(f"Loaded {len(docs)} documents")
# Use local sentence transformer for embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db1 = FAISS.from_documents(docs, embedding)
print("Vector DB size:", len(db1.index_to_docstore_id))

vector = FAISS.from_documents(docs, embedding)
vector.save_local("faiss_db")
db2 = FAISS.load_local("faiss_db",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

print("Vector DB size:", len(db2.index_to_docstore_id))

# Save the vector index
db1.save_local("crop_faiss_db1_chunk")
db2.save_local("crop_faiss_db2_chunk")