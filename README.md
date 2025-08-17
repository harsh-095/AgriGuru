## AgriGuru

Progress

To-Do

1. Query Expansion - To have higher hit rate
2. Multi-Retrieval (1st level- ANN:approx nearest, 2nd level - Finer )
3. Memory For Resonse
4. Cache for responses
5. Context Expansion
6. Text, Image, Audio , Video Embeddings
7. Performance Optimization
8. Use OpenRoute or Mistral apis
9. Query based searchs , SQL query integration
   10 . Optimization Techniques like User Feedback
10. Invalidating wrong response in cache or remove old data

#### Set Up

Ref: https://github.com/whyashthakker/RAG

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
