from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import faiss
import numpy as np
import pickle
from transformers import CLIPProcessor

model = SentenceTransformer("clip-ViT-B-32", use_fast=True)

def embed_image(path):
    image = Image.open(path).convert("RGB")
    return model.encode([image])[0]


base_path = "imageDB/image data/train_1"
vectors, metadatas = [], []

for crop in os.listdir(base_path):
    crop_path = os.path.join(base_path, crop)
    if not os.path.isdir(crop_path): continue

    for label in os.listdir(crop_path):
        label_path = os.path.join(crop_path, label)
        if not os.path.isdir(label_path): continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            vec = embed_image(img_path)
            vectors.append(vec)
            metadatas.append({
                "crop": crop,
                "label": label,
                "status": "Healthy" if "healthy" in label.lower() else "Diseased",
                "path": img_path
            })

# Save FAISS + Metadata
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors))

faiss.write_index(index, "img_resources/disease_faiss.index")
with open("img_resources/disease_meta.pkl", "wb") as f:
    pickle.dump(metadatas, f)
