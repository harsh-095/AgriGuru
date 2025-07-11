# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Run 
# uvicorn ui_be:app --reload

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TextMsg(BaseModel):
    message: str

@app.post("/end1")
@app.post("/end3")
def text_bot(data: TextMsg):
    return {"response": f"Echo from {'/end1' if '/end1' in str(app.router.routes[0].path) else '/end3'}: {data.message}"}

@app.post("/end2")
async def image_bot(image: UploadFile = File(...)):
    return f"Image {image.filename} received successfully!"
