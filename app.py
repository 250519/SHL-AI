import tempfile
import gradio as gr
from core.llm_processor import preprocess_input
from core.retrieval import retrieve_and_rerank
import pandas as pd

# === Test Type Decoder ===
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

def format_results(raw_results):
    formatted = []
    for r in raw_results:
        formatted.append({
            "Test Name": r.get("Test Name", ""),
            "URL": r.get("Test Link", ""),
            "Description": r.get("Description", ""),
            "Duration": r.get("Assessment Length", ""),
            "Remote Support": r.get("Remote Support") if r.get("Remote Support") else "No",
            "Adaptive Support": r.get("Adaptive Support") if r.get("Adaptive Support") else "No",
            "Test Type": decode_test_type(r.get("Test Type", "")),
        })
    return pd.DataFrame(formatted)

# === Gradio Interface Logic ===
def format_results_for_display(raw_results):
    return pd.DataFrame(raw_results)

def recommend_with_download(query_input: str):
    try:
        print("📥 Received:", query_input)
        refined_query = preprocess_input(query_input)
        print("🔍 Refined:", refined_query)

        results = retrieve_and_rerank(refined_query)
        if not results:
            return "No results found", pd.DataFrame(), None

        df = format_results(results)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        return f"{len(df)} results found.", df, temp_file.name

    except Exception as e:
        print("❌ Error:", e)
        return str(e), pd.DataFrame(), None

# === Gradio UI ===
gr.Interface(
    fn=recommend_with_download,
    inputs=gr.Textbox(label="Enter your query, job description, or URL"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Dataframe(label="Top 10 Recommended Assessments", wrap=True),
        gr.File(label="📥 Download CSV")
    ],
    title="SHL Assessment Recommender",
    description="Paste a natural language query, a full JD, or a JD URL to get the top 10 assessment matches.",
    allow_flagging="never"
).launch()
