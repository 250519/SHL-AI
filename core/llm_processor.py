import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

def is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")

def is_probable_jd(text: str) -> bool:
    return (
        len(text.split()) > 50 or (
            "responsibilities" in text.lower() or
            "qualifications" in text.lower() or
            "job description" in text.lower() or
            "apply now" in text.lower() or
            "skills required" in text.lower()
        )
    )

def extract_jd_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        candidates = soup.find_all(["p", "div", "section", "article"], recursive=True)
        jd_candidates = [c.get_text(strip=True, separator=" ") for c in candidates if len(c.get_text(strip=True)) > 100]
        jd_text = "\n".join(jd_candidates[:5])  # Limit to top 5 blocks
        return jd_text
    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return ""

def llm_extract_query_from_jd(jd_text: str) -> str:
    prompt = f"""
You are an intelligent assistant that converts job descriptions into smart search queries to find suitable assessment for this JD.

Your job is to read the JD(Job Description) below and write a concise query that captures:
- The role or domain
- Hard skills
- Soft skills or traits (like communication, collaboration)
- Any constraints like seniority level or duration
- Combine these into one smart search query

It should be sentence-like, not a list. Use natural language.
For Example:"I am hiring for Java developers who can also collaborate effectively with my business teams. Looking 
for an assessment(s) that can be completed in 40 minutes."

Return ONLY the final query string. No commentary or explanations.


Job Description:
{jd_text}
"""
    try:
        print("⏳ Calling Gemini...")
        response = llm.generate_content(prompt)
        # print("response",response)
        response_text = response.text.strip()
        print(" Gemini returned:\n", response_text)

        return response.text.strip()
    except Exception as e:
        print(f" LLM error: {e}")
        return ""

def preprocess_input(user_input: str) -> str:
    if is_url(user_input):
        print("  Detected URL input — scraping JD...")
        jd_text = extract_jd_from_url(user_input)
        if not jd_text:
            return "Could not extract job description from URL."
        return llm_extract_query_from_jd(jd_text)

    elif is_probable_jd(user_input):
        print(" Detected JD text — parsing with LLM...")
        return llm_extract_query_from_jd(user_input)

    else:
        print(" Detected simple query — using as-is.")
        return user_input.strip()
