from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm

# Load model
model = SentenceTransformer("clip-ViT-B-32")

# Path setup
base_path = "imageDB/image data/train"
image_paths = []
metadata_list = []

# Scan all images and prepare metadata
for crop in os.listdir(base_path):
    crop_path = os.path.join(base_path, crop)
    if not os.path.isdir(crop_path): continue

    for label in os.listdir(crop_path):
        label_path = os.path.join(crop_path, label)
        if not os.path.isdir(label_path): continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image_paths.append(img_path)
            metadata_list.append({
                "crop": crop,
                "label": label,
                "status": "Healthy" if "healthy" in label.lower() else "Diseased",
                "path": img_path
            })

# ✅ Batch embed
BATCH_SIZE = 32
embeddings = []

print(f"📸 Found {len(image_paths)} images. Embedding...")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    images = []
    valid_meta = []

    for j, path in enumerate(batch_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_meta.append(metadata_list[i + j])
        except Exception as e:
            print(f"⚠️ Skipped {path}: {e}")

    if images:
        embs = model.encode(images, convert_to_numpy=True, batch_size=BATCH_SIZE)
        embeddings.extend(embs)
        metadata_list[i:i+len(valid_meta)] = valid_meta  # keep only valid metadata

# ✅ Save FAISS index + metadata
if embeddings:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs("img_resources/db_all_train", exist_ok=True)
    faiss.write_index(index, "img_resources/db_all_train/disease_faiss.index")

    with open("img_resources/db_all_train/disease_meta.pkl", "wb") as f:
        pickle.dump(metadata_list[:len(embeddings)], f)

    print(f"\n✅ Done. Saved {len(embeddings)} image vectors.")
else:
    print("❌ No embeddings created.")
