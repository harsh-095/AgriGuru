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
from test_001_DB import ask_sql
from fastapi.encoders import jsonable_encoder

# Run using - 
# streamlit run test_001_FE.py

app = FastAPI()

class QueryInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(input: QueryInput):
    print("Received Question:" + input.question)
    response = ask_sql(input.question)

    # 🔥 Convert tuples to lists to avoid serialization issues
    if "SQL_Result" in response and isinstance(response["SQL_Result"], list):
        response["SQL_Result"] = [list(row) for row in response["SQL_Result"]]

    print("Response=")
    print(response)

    # ✅ Make it JSON safe
    safe_response = jsonable_encoder(response)
    return {"Response": safe_response}
