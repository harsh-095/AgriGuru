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

print(f"Loaded {len(docs)} documents")

# Use local sentence transformer for embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)

# Save the vector index
db.save_local("crop_faiss_db")



# Load DB and set up retriever

db = FAISS.load_local(
    folder_path=r"E:\InfyHackathon\crop_faiss_db",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# Connect to local Ollama model
llm = Ollama(model="deepseek-r1:1.5b")  # or llama3

# Build RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Ask a question
query = "Suggest a crop for high humidity and acidic soil"
response = qa.run(query)
print("Bot says:", response)
