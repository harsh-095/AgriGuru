from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


#  Run Api 
# uvicorn crop_api:app --reload --host 0.0.0.0 --port 8000



# 🔁 Setup once
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    folder_path="crop_faiss_db1",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 2200})
llm = Ollama(model="gemma3:4b")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🚀 FastAPI app
app = FastAPI()

class QueryInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(input: QueryInput):
    response = qa.invoke({"query": input.question + " based on dataset provided"})
    return {"answer": response}

@app.get("/")
def root():
    return {"message": "Crop Recommendation AI is running"}
