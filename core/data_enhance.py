import pandas as pd
import json
import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
df = pd.read_csv("shl_enhanced_assessments_clean.csv")

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"temperature": 0.9})

def generate_prompt(title: str, description: str) -> str:
    return f"""
You are an intelligent assistant designed to improve AI-based search and recommendation of assessments.

Your task is to analyze the given assessment title and description, and extract a concise list of relevant tags. These tags will help match assessments to user queries like:
“I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.”

Extract tags that accurately reflect:

Hard skills (e.g., Java, SQL, REST APIs)

Soft skills (e.g., collaboration, communication)

Job roles or levels (e.g., backend developer, mid-level, team lead)

Business domain or context (e.g., cross-functional, agile)

Assessment traits (e.g., time limit, difficulty level, scenario-based)

Be specific and comprehensive. Avoid generic or vague tags like “test” or “assessment.” Use domain-relevant terminology.

Return output ONLY in this JSON format:
{{"tags": ["tag1", "tag2", "tag3"]}}

Title: "{title}"
Description: "{description}"
"""

tag_column = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    title = str(row.get("Test Name", ""))
    description = str(row.get("Description", ""))

    prompt = generate_prompt(title, description)

    try:
        response = llm.generate_content(prompt)
        response_text = response.text.strip()

        response_text = re.sub(r"^```json", "", response_text)
        response_text = re.sub(r"```$", "", response_text)
        response_text = response_text.strip()

        parsed = json.loads(response_text)
        tags = parsed.get("tags", [])
    except Exception as e:
        tags = []
        print(f"Error on '{title}': {e}")

    tag_column.append(tags)