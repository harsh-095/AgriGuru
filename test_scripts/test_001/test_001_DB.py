import pandas as pd
import sqlite3
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from mistralai import Mistral
from sqlalchemy import text
from getpass import getpass
import os

# Run using  
# uvicorn test_001_BE:app --reload --host 0.0.0.0 --port 8000

client = Mistral(api_key="API_KEY")
model = "mistral-small-2506"

base_path = os.path.dirname(os.path.abspath(__file__))

# construct path to csv inside resources
csv_path = os.path.join(base_path, "resources", "Crop_recommendation.csv")
db_path = f"""sqlite:///{base_path}/resources/crops.db"""

print("CSV Path:", csv_path)
print("db Path:", db_path)

df = pd.read_csv(csv_path)
print(df.head())

# Save to SQLite
engine = create_engine(db_path)
df.to_sql("crops", con=engine, if_exists="replace", index=False)
db = SQLDatabase(engine)

def extract_sql(answer: str) -> str:
    """
    Extract SQL code block or first SELECT statement from LLM output.
    """
    # If fenced code block
    match = re.search(r"```sql\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Otherwise, try to find the first SELECT
    match = re.search(r"(SELECT .*?;)", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # fallback (return as-is, might fail)
    return answer.strip()


def ask_sql(query: str):
    prompt = f"""
    You are a data assistant.
    The table name is 'crops'. Schema: {df.dtypes.to_dict()}

    Note: 
    1. Convert label value to lower case and correct spellings if required before querying for label column
    2. Never write Select * in query as there might be serveral records for same label , so if no much details are provided, use aggregate functions
    ONLY return a valid SQLite SQL query, no explanations.
    Query: "{query}"
    """
    response = client.chat.complete(
        model="mistral-small-2506",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content.strip()
    sql_query = extract_sql(raw_output)

    # Run SQL
    with engine.connect() as conn:
        result = conn.execute(text(sql_query)).fetchall()
        explanation_prompt = f"""
    You are a helpful data assistant. 
    The user asked: "{query}"
    The SQL Query Generated: {sql_query}
    The SQL result is: {result}
    
    Please answer the question in natural language based on the result.
    """
    explanation = client.chat.complete(
        model="mistral-small",
        messages=[{"role": "user", "content": explanation_prompt}]
    )

    answer_sentence = explanation.choices[0].message.content.strip()
    return {"Question":query,"SQL_Query":sql_query,"SQL_Result": result,"Answer": answer_sentence}
