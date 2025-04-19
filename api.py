from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from core.llm_processor import preprocess_input
from core.retrieval import retrieve_and_rerank

app = FastAPI()

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}

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

# === 3. Request schema ===
class RecommendationRequest(BaseModel):
    input: str

# === 4. Response schema ===
class Assessment(BaseModel):
    Test_Name: str
    URL: str
    Description: str
    Duration: str
    Remote_Support: str
    Adaptive_Support: str
    Test_Type: str

# === 5. Main Recommend Endpoint ===
@app.post("/recommend", response_model=List[Assessment])
def recommend_assessments(payload: RecommendationRequest):
    try:
        if not payload.input or not payload.input.strip():
            raise HTTPException(status_code=400, detail="Input cannot be empty.")

        refined_query = preprocess_input(payload.input)
        results = retrieve_and_rerank(refined_query)

        if not results:
            raise HTTPException(status_code=404, detail="No relevant assessments found.")

        return [
            {
                "Test_Name": r.get("Test Name", ""),
                "URL": r.get("Test Link", ""),
                "Description": r.get("Description", ""),
                "Duration": r.get("Assessment Length", ""),
                "Remote_Support": r.get("Remote Support") or "No",
                "Adaptive_Support": r.get("Adaptive Support") or "No",
                "Test_Type": decode_test_type(r.get("Test Type", "")),
            }
            for r in results[:10]
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
