import json
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

try:
    from google.api_core.exceptions import (
        GoogleAPICallError,
        InvalidArgument,
        ResourceExhausted,
        Unauthenticated,
    )
except Exception:  # pragma: no cover
    GoogleAPICallError = Exception
    InvalidArgument = Exception
    ResourceExhausted = Exception
    Unauthenticated = Exception


load_dotenv()

gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
gemini_ready = bool(gemini_api_key and genai is not None)

if gemini_ready:
    genai.configure(api_key=gemini_api_key)


@st.cache_data(ttl=3600)
def _list_gemini_models() -> list[str]:
    if not gemini_ready:
        return []
    try:
        models: list[str] = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" in methods and getattr(m, "name", None):
                models.append(m.name)
        return sorted(set(models))
    except Exception:
        return []


def _normalize_model_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    if raw.startswith("models/"):
        return raw
    return f"models/{raw}"


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", text or "")
    return [t.lower() for t in tokens]


def _local_resume_analysis(resume_text: str, jd_text: str) -> tuple[str, dict]:
    resume_tokens = set(_tokenize(resume_text))
    jd_tokens = set(_tokenize(jd_text))

    common = sorted(resume_tokens & jd_tokens) if jd_tokens else []
    common_top = common[:25]

    word_count = len((resume_text or "").split())
    has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text or ""))
    has_phone = bool(re.search(r"(\+?\d[\d\s().-]{7,}\d)", resume_text or ""))

    skills_match = 0
    if jd_tokens:
        overlap_ratio = len(resume_tokens & jd_tokens) / max(1, len(jd_tokens))
        skills_match = int(round(min(30, overlap_ratio * 60)))

    experience_achievements = 20
    if re.search(r"\b(\d{4})\b", resume_text or ""):
        experience_achievements += 5
    if re.search(r"\b(led|built|launched|improved|increased|reduced|optimized)\b", resume_text or "", re.I):
        experience_achievements += 5
    experience_achievements = min(30, experience_achievements)

    clarity_formatting = 10
    if word_count >= 150:
        clarity_formatting += 5
    if has_email and has_phone:
        clarity_formatting += 5
    clarity_formatting = min(20, clarity_formatting)

    overall_impression = 12
    if re.search(r"\b(projects?|experience|education|skills?)\b", resume_text or "", re.I):
        overall_impression += 4
    if common_top:
        overall_impression += 4
    overall_impression = min(20, overall_impression)

    total = int(skills_match + experience_achievements + clarity_formatting + overall_impression)
    score = {
        "skills_match": int(skills_match),
        "experience_achievements": int(experience_achievements),
        "clarity_formatting": int(clarity_formatting),
        "overall_impression": int(overall_impression),
        "total": int(min(100, max(0, total))),
    }

    analysis = "\n".join(
        [
            "Summary:",
            f"- Word count: {word_count}",
            f"- Contact info detected: email={'yes' if has_email else 'no'}, phone={'yes' if has_phone else 'no'}",
            "",
            "Key Skills:",
            ("- Matching JD keywords: " + ", ".join(common_top)) if jd_tokens else "- Provide a job description to compute keyword match.",
            "",
            "Suggestions for Improvement:",
            "- Add measurable impact (numbers, % improvements) where possible.",
            "- Ensure a clear Skills section aligned to the job description.",
            "- Keep bullet points concise and action-oriented.",
        ]
    )

    return analysis, score


def _format_output_with_json(analysis_text: str, score_data: dict) -> str:
    try:
        json_line = json.dumps(score_data, ensure_ascii=False)
    except Exception:
        json_line = str(score_data)
    return (analysis_text or "").rstrip() + "\n" + json_line


def _build_ats_prompt(job_description: str, resume_text: str) -> str:
    text_clean = (resume_text or "").strip()
    jd = (job_description or "").strip()

    return f"""You are an expert ATS resume evaluator and career coach.

TASK:
Analyze the given resume against the provided job description (if any).

OUTPUT REQUIREMENTS (VERY IMPORTANT):
1. First, write a clear human-readable analysis with these exact sections:
     - Summary
     - Key Skills (bullet points)
     - Suggestions for Improvement (bullet points)

2. AFTER the analysis, output ONE JSON object on a NEW LINE.
3. The JSON MUST be the LAST thing in your response.
4. The JSON MUST be valid (no markdown, no comments, no trailing commas).
5. All values MUST be integers.
6. Do NOT add any extra text after the JSON.

SCORING RULES:
- skills_match: 0–30 (match between resume skills and job description)
- experience_achievements: 0–30 (projects, impact, leadership, numbers)
- clarity_formatting: 0–20 (structure, readability, contact info)
- overall_impression: 0–20 (professional quality)
- total: sum of all above (0–100)

JSON SCHEMA (FOLLOW EXACTLY):
{{
    "skills_match": 0,
    "experience_achievements": 0,
    "clarity_formatting": 0,
    "overall_impression": 0,
    "total": 0
}}

JOB DESCRIPTION:
{jd}

RESUME:
{text_clean}
"""


def _parse_trailing_json(text: str) -> dict | None:
    if not (text or "").strip():
        return None
    match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _render_score(score_data: dict) -> None:
    st.subheader("Resume score breakdown")
    st.json(score_data)

    df = pd.DataFrame(
        {
            "category": [
                "skills_match",
                "experience_achievements",
                "clarity_formatting",
                "overall_impression",
            ],
            "score": [
                score_data.get("skills_match", 0),
                score_data.get("experience_achievements", 0),
                score_data.get("clarity_formatting", 0),
                score_data.get("overall_impression", 0),
            ],
        }
    )
    st.bar_chart(df.set_index("category"))

    total = float(score_data.get("total", 0) or 0)
    st.subheader("Overall score")
    st.progress(max(0.0, min(1.0, total / 100.0)))


st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="wide")
st.title("AI Resume Analyzer")
st.markdown(
    "Upload your resume and job description to get an AI-powered analysis of your resume's effectiveness for the specified job role."
)

uploaded_resume = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("Job description (optional)", height=200)

st.sidebar.header("Settings")
use_gemini = st.sidebar.checkbox("Use Gemini", value=gemini_ready)

if use_gemini and not gemini_ready:
    st.sidebar.warning("Gemini not ready: set GEMINI_API_KEY (and install google-generativeai).")

env_model = _normalize_model_name(os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash"))
available_models = _list_gemini_models() if gemini_ready else []

if gemini_ready and available_models:
    default_index = available_models.index(env_model) if env_model in available_models else 0
    model_name = st.sidebar.selectbox("Gemini model", options=available_models, index=default_index)
elif gemini_ready:
    model_name = _normalize_model_name(st.sidebar.text_input("Gemini model", value=env_model))
else:
    model_name = env_model

if not gemini_api_key:
    st.info("Set GEMINI_API_KEY in your .env to enable Gemini analysis (offline mode still works).")


def _extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)
    text = "\n".join(parts)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


if uploaded_resume is not None:
    text_clean = _extract_pdf_text(uploaded_resume)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Resume preview")
        st.text_area("Resume text", value=text_clean, height=420)

    with col2:
        st.subheader("Analysis")
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                if not use_gemini or not gemini_ready:
                    analysis, score_data = _local_resume_analysis(text_clean, job_description)
                    st.text_area("Model output", value=_format_output_with_json(analysis, score_data), height=350)
                    _render_score(score_data)
                    st.stop()

                prompt = _build_ats_prompt(job_description, text_clean)
                try:
                    if available_models and model_name not in available_models:
                        raise InvalidArgument(
                            f"Model '{model_name}' is not available for this API key. Pick one from the dropdown."
                        )

                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config={"max_output_tokens": 1400},
                    )
                    result = getattr(response, "text", "") or ""

                    if not result.strip():
                        st.warning(
                            "Gemini returned no text output. This can happen with quota=0, "
                            "safety filters, or empty candidates."
                        )
                    st.text_area("Model output", value=result, height=350)

                except ResourceExhausted as e:
                    st.error(f"Gemini quota/rate limit error: {e}. Falling back to offline analysis.")
                    analysis, score_data = _local_resume_analysis(text_clean, job_description)
                    st.text_area("Model output", value=_format_output_with_json(analysis, score_data), height=350)
                    _render_score(score_data)
                    st.stop()
                except Unauthenticated as e:
                    st.error(f"Gemini authentication failed. Check GEMINI_API_KEY. Details: {e}")
                    analysis, score_data = _local_resume_analysis(text_clean, job_description)
                    st.text_area("Model output", value=_format_output_with_json(analysis, score_data), height=350)
                    _render_score(score_data)
                    st.stop()
                except (InvalidArgument, GoogleAPICallError, Exception) as e:
                    st.error(f"Gemini request failed: {e}. Falling back to offline analysis.")
                    analysis, score_data = _local_resume_analysis(text_clean, job_description)
                    st.text_area("Model output", value=_format_output_with_json(analysis, score_data), height=350)
                    _render_score(score_data)
                    st.stop()

            score_data = _parse_trailing_json(result)
            if score_data:
                _render_score(score_data)
            else:
                st.warning("Could not parse trailing JSON from the model output.")
                        