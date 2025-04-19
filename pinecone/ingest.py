import os
import time
import ast
import pandas as pd
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "shl"
embedding_dim = 384

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="enabled"
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

df = pd.read_csv("data/shl_enhanced_assessments_with_tags.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

batch_size = 25
vectors_to_upsert = []

def parse_tags(tag_value):
    try:
        tags = ast.literal_eval(tag_value) if isinstance(tag_value, str) else tag_value
        return tags if isinstance(tags, list) else []
    except:
        return []

for i, row in df.iterrows():
    row = row.fillna("")
    text = f"{row['Test Name']}. {row['Description']}"
    embedding = model.encode(text).tolist()

    metadata = {
        "Test Name": str(row["Test Name"]),
        "Test Link": str(row["Test Link"]),
        "Description": str(row["Description"]),
        "Assessment Length": str(row["Assessment Length"]),
        "Job Levels": str(row["Job Levels"]),
        "Remote Testing": str(row["Remote Testing"]),
        "Adaptive/IRT": str(row["Adaptive/IRT"]),
        "Test Type": str(row["Test Type"]),
        "Tags": parse_tags(row["Tags"]),
    }

    vector = (str(uuid4()), embedding, metadata)
    vectors_to_upsert.append(vector)

    if len(vectors_to_upsert) == batch_size or i == len(df) - 1:
        print(f"🔼 Upserting batch {i + 1 - batch_size + 1} to {i + 1}")
        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"Error during upsert: {e}")
            print("⏳ Retrying by reconnecting...")
            try:
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                index = pc.Index(index_name)
                index.upsert(vectors=vectors_to_upsert)
                print("Retry succeeded.")
            except Exception as inner_e:
                print(f"Retry failed again: {inner_e}")
                break
        vectors_to_upsert = []
