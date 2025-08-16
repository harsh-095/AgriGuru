## AgriGuru

#### Set Up

```
pip install streamlit requests
pip install python-multipart
pip install sentence-transformers torchvision pillow numpy
pip install fastapi uvicorn
pip install -U langchain-huggingface
pip install -U langchain-community
pip install faiss-cpu langchain sentence-transformers pandas
```

Files:
BE : full_be_api.py
UI : ui_fe.py

Used Models

```
For Image to Text Embedding: For Index Creation: clip-ViT-B-32

Crop Recommendation
For Embedding: sentence-transformers/all-MiniLM-L6-v2

General Model: gemma3:1b
```

Run Commands:

# UI

Run Using
streamlit run ui_fe.py

# BE

uvicorn full_be_api:app --reload --host 0.0.0.0 --port 8000
