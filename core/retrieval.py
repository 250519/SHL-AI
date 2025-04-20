import os
import re
import time
import json
from typing import List, Dict, Tuple
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
def get_index():
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return pc.Index("shl")
    except Exception as e:
        print(f" Pinecone init failed: {e}")
        print("⏳ Retrying in 3s...")
        time.sleep(3)
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            return pc.Index("shl")
        except Exception as inner_e:
            print(f"Retry failed: {inner_e}")
            raise inner_e
        

index = get_index()

model = SentenceTransformer("all-MiniLM-L6-v2")

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

def decode_test_type(code: str) -> str:
    return ", ".join(TEST_TYPE_MAP.get(c.strip(), c) for c in code if c.strip() in TEST_TYPE_MAP)


def build_prompt(query: str, blocks: List[str]) -> str:
    return f"""
You are an expert assistant helping HR teams and recruiters select the most relevant assessments for their hiring needs.

Given a natural language hiring query or Job description of a job and a list of available assessments, your task is to intelligently rank and recommend the top 10 most relevant assessments.

Assessments are described using their title, description, tags, target job level, and duration. The user query may include specific technical and soft skills, job roles, team collaboration needs, or constraints like duration (e.g., "within 40 minutes").

You must:
- Match skills (technical and soft)
- Understand job role and seniority
- Respect duration constraints
- Recognize contextual needs (e.g., cross-functional, remote-friendly, leadership focus)
- Rank results based on semantic relevance, not just keyword overlap
-Also respect the **Test Type**, where each letter has a specific meaning:
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
    
Return a JSON list of objects like:
[
  {{ "id": "3", "reason": "Matches Java, collaboration, and duration" }},
  ...
]

Query:
"{query}"

Assessments:
{''.join(blocks)}
"""

def retrieve_and_rerank(query: str, top_k: int = 50) -> List[Dict]:
    query_vector = model.encode(query).tolist()

    # Step 1: Retrieve from Pinecone
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Step 2: Prepare blocks for reranking
    id_map = {}
    assessment_blocks = []

    for i, match in enumerate(response['matches']):
        md = match['metadata']
        aid = str(i + 1)
        id_map[aid] = md

        tags = ", ".join(md.get("Tags", [])) if isinstance(md.get("Tags"), list) else md.get("Tags", "")
        test_type_mapped = decode_test_type(md.get("Test Type", ""))

        block = f"""{aid}. Title: {md.get('Test Name', '')}
    Description: {md.get('Description', '')}
    Tags: {tags}
    Job Level: {md.get('Job Levels', '')}
    Duration: {md.get('Assessment Length', '')}
    Test Type: {test_type_mapped}
    """

        assessment_blocks.append(block)

    # Step 3: Build prompt & rerank
    prompt = build_prompt(query, assessment_blocks)
    final_results = []
    reranked = []

    try:
        print(" Gemini rerank started...")
        response = llm.generate_content(prompt)
        response_text = response.text.strip()

        # Cleanup code blocks
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"```$", "", response_text).strip()

        print("🧾 Gemini raw output (trimmed):\n", response_text[:1000])
        reranked = json.loads(response_text)
        print(" Parsed reranked list:", reranked)

    except Exception as e:
        print(f"Failed to rerank: {e}")
        print(f"Raw response (if available):\n{locals().get('response_text', 'N/A')}")
        reranked = []

    # Step 4: Parse reranked results
    for item in reranked:
        aid = str(item.get("id", "")).strip().rstrip(".")
        reason = item.get("reason", "No reason given")
        md = id_map.get(aid)

        if not md:
            print(f" ID {aid} not found in id_map — skipping.")
            continue

        final_results.append({
            "Test Name": md.get("Test Name"),
            "Test Link": md.get("Test Link"),
            "Description": md.get("Description"),
            "Assessment Length": md.get("Assessment Length"),
            "Remote Support": md.get("Remote Testing") or "No",
            "Adaptive Support": md.get("Adaptive/IRT") or "No",
            "Test Type": md.get("Test Type"),
            "Reason": reason
        })

    print("Final results count:", len(final_results))
    return final_results
