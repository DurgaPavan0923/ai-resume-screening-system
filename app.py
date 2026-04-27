import streamlit as st
import pandas as pd
import plotly.express as px
import base64

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_roles
from src.experience_extractor import extract_experience
from src.education_parser import extract_education
from src.highlighter import highlight_text

from utils.helpers import validate_input, format_skills
from config import SKILLS_PATH


# =========================
# PDF VIEWER
# =========================
def show_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px"></iframe>'


# =========================
# CSS (UNCHANGED)
# =========================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&display=swap');

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        color: #00e6ff !important;
    }

    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
    }

    .card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Resume Dashboard", layout="wide")
load_css()


# =========================
# MODEL
# =========================
@st.cache_resource
def get_model():
    return train_model()

model, vectorizer = get_model()


# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI</div>', unsafe_allow_html=True)


# =========================
# INPUT
# =========================
job_desc = st.text_area("Job Description")
files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)


# =========================
# ANALYZE
# =========================
if st.button("Analyze Candidates"):

    skills_db = load_skills(SKILLS_PATH)
    job_clean = clean_text(job_desc)

    results = []
    raw_texts = {}

    for file in files:
        text = parse_pdf(file)

        raw_texts[file.name] = text
        file.seek(0)
        raw_texts[file.name + "_file"] = file

        clean = clean_text(text)

        similarity = compute_similarity(job_clean, clean, vectorizer)
        skills = extract_skills(clean, skills_db) or {}
        jd_skills = extract_skills(job_clean, skills_db) or {}

        experience = extract_experience(clean)
        education = extract_education(clean)

        roles, ml_roles = predict_roles(clean, skills, model, vectorizer)

        try:
            from src.gpt_analyzer import analyze_resume
            gpt_analysis = analyze_resume(text, job_desc)
        except:
            gpt_analysis = "AI analysis unavailable"

        score = round(similarity * 100, 2)

        results.append({
            "name": file.name,
            "score": score,
            "skills": skills,
            "roles": roles,
            "ml_roles": ml_roles,
            "experience": experience,
            "gpt_analysis": gpt_analysis,
            "missing_skills": list(set(jd_skills) - set(skills))
        })

    # =========================
    # DISPLAY
    # =========================
    for r in results:

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        # COLUMN 1 → ANALYSIS
        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>{r['name']}</h4>
                <p>Score: {r['score']}%</p>
                <p>Roles: {', '.join(r['roles'])}</p>
                <p>Skills: {format_skills(r['skills'])}</p>
                <p>Experience: {r['experience']} yrs</p>
            </div>
            """, unsafe_allow_html=True)

        # COLUMN 2 → PDF PREVIEW
        with col2:
            file_obj = raw_texts.get(r["name"] + "_file")
            if file_obj:
                file_obj.seek(0)
                st.markdown(show_pdf(file_obj), unsafe_allow_html=True)

        # COLUMN 3 → INSIGHTS
        with col3:

            with st.expander("Role Confidence"):
                for role, sc in r["ml_roles"]:
                    st.write(f"{role}: {sc}%")

            with st.expander("Skill Gap"):
                for s in r["missing_skills"]:
                    st.write(f"❌ {s}")

            with st.expander("AI Analysis"):
                st.write(r["gpt_analysis"])

            with st.expander("Resume Highlight"):
                highlighted = highlight_text(raw_texts[r["name"]], list(r["skills"].keys()))
                st.markdown(highlighted, unsafe_allow_html=True)

    # =========================
    # SUMMARY (NEW)
    # =========================
    if results:
        top = max(results, key=lambda x: x["score"])

        st.markdown("---")
        st.subheader("Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Top Candidate", top["name"])
        c2.metric("Score", f"{top['score']}%")
        c3.metric("Roles", ", ".join(top["roles"]))


# =========================
# FOOTER (NEW)
# =========================
st.markdown("---")
st.markdown("Advanced AI Resume Screening System")