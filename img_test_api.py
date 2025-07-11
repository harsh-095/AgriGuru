from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import pickle
import os

app = FastAPI()

# uvicorn img_test_api:app --reload --host 0.0.0.0 --port 8000

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
