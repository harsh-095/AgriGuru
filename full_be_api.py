from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import pickle
import os
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


#  Run Api 
# uvicorn full_be_api:app --reload --host 0.0.0.0 --port 8000



# 🔁 Setup once
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    folder_path="mmr_crop_faiss_db_001",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 10}
# )
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 50, "lambda_mult": 0.3}
)


llm = Ollama(model="gemma3:1b")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🚀 FastAPI app
app = FastAPI()

model = SentenceTransformer("clip-ViT-B-32")

def embed_image(path):
    image = Image.open(path).convert("RGB")
    return model.encode([image])[0]

def predict_disease(uploaded_path, k=1):
    vec = embed_image(uploaded_path).reshape(1, -1)
    index = faiss.read_index("img_resources/db_all_train/disease_faiss.index")
    with open("img_resources/db_all_train/disease_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    _, indices = index.search(vec, k)
    return [meta[i] for i in indices[0]]


@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = predict_disease(temp_path)
    return JSONResponse(content={"diagnosis": results})

class QueryInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(input: QueryInput):
    response = qa.invoke({"query": input.question + " based on dataset provided"})
    return {"answer": response}

@app.get("/")
def root():
    return {"message": "Crop Recommendation AI is running"}
