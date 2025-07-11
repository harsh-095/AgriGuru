import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS






df = pd.read_csv("resources\Crop_recommendation.csv")
print("read_csv completed")

docs = []

for _, row in df.iterrows():
    content = f"""Nitrogen in soil (N): {row['N']}, Phosphorous in soil (P): {row['P']}, Potassium in soil (K): {row['K']}, Temperature of Air or Weather Temperature: {row['temperature']}, Humidity in air or Atmosphere Humidity : {row['humidity']}, 
pH level of soil or measure of how acidic(low pH - 0-6.5)/ basic(High pH - 7.5-14)/ neutral(6.5-7.5) the soil is: {row['ph']}, Rainfall in that area: {row['rainfall']} — Recommended Crop: {row['label']}"""
    doc = Document(page_content=content, metadata={"crop": row["label"]})
    docs.append(doc)


print("docs.append completed")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



vector = FAISS.from_documents(docs, embedding)
vector.save_local("mmr_crop_faiss_db_001")
print("save_local completed")