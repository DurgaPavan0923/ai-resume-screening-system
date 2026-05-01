import streamlit as st
import pandas as pd
import plotly.express as px

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
# CSS
# =========================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&family=Inter:wght@400;500&display=swap');

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        color: #00e6ff !important;
        text-shadow: 0px 0px 20px rgba(0,255,255,0.8);
    }

    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        color: #010c0d !important;
    }

    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #00e6ff !important;
    }

    :root {
        --text-color: #111;
        --card-bg: rgba(255,255,255,0.6);
        --border-color: rgba(0,0,0,0.1);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #fff;
            --card-bg: rgba(255,255,255,0.08);
            --border-color: rgba(255,255,255,0.2);
        }
    }

    .card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        margin: 12px;
        border: 1px solid var(--border-color);
        color: var(--text-color);
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
# HELPERS
# =========================
def skill_match_score(jd_skills, resume_skills, skills_db):
    if not jd_skills:
        return 0
    total = sum(skills_db.get(s, 1) for s in jd_skills)
    matched = sum(skills_db.get(s, 1) for s in resume_skills if s in jd_skills)
    return matched / total if total else 0


def get_decision(score):
    if score >= 75:
        return "<span style='color:lime'>🟢 Hire</span>"
    elif score >= 50:
        return "<span style='color:orange'>🟡 Consider</span>"
    else:
        return "<span style='color:red'>🔴 Reject</span>"


def skill_gap(jd_skills, resume_skills):
    jd_set = set(jd_skills.keys() if isinstance(jd_skills, dict) else jd_skills)
    resume_set = set(resume_skills.keys() if isinstance(resume_skills, dict) else resume_skills)
    return list(jd_set - resume_set)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("Dashboard")
    st.write("AI Resume Analyzer")
    st.markdown("### Tips")
    st.write("✔ Use detailed job descriptions")
    st.write("✔ Add skills")
    st.write("✔ Upload multiple resumes")


# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# =========================
# INPUT
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("Job Description", height=200)

with col2:
    files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)


# =========================
# RANKED CANDIDATES (UPDATED)
# =========================
st.subheader("Ranked Candidates")

cols = st.columns(2)

for i, r in enumerate(results):
    with cols[i % 2]:

        # ===== ORIGINAL CARD (UNCHANGED) =====
        st.markdown(f"""
        <div class="card">
            <h3>{r['name']}</h3>
            <p>{r['decision']}</p>
            <p>Score: {r['score']}%</p>
            <p><b>Roles:</b> {r['role']}</p>
            <p>Skills: {format_skills(r['skills'])}</p>
            <p>Experience: {r['experience']} years</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(r["score"] / 100)

        # =========================
        # 🔥 NEW 3-COLUMN SECTION
        # =========================
        c1, c2, c3 = st.columns([1.1, 1.5, 1.2])

        # =========================
        # COLUMN 1 → ANALYSIS DETAILS
        # =========================
        with c1:
            st.markdown("### 📊 Analysis")
            st.write(f"**Education:** {', '.join(r['education']) if r['education'] else 'Not detected'}")

        # =========================
        # COLUMN 2 → PDF PREVIEW
        # =========================
        with c2:
            st.markdown("### 📄 Resume Preview")

            file_obj = raw_texts.get(r["name"] + "_file")
            if file_obj:
                file_obj.seek(0)
                base64_pdf = base64.b64encode(file_obj.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.warning("Preview not available")

        # =========================
        # COLUMN 3 → INSIGHTS PANEL
        # =========================
        with c3:

            # Role Confidence
            with st.expander("Role Confidence"):
                for role_name, score in r["ml_roles"]:
                    st.write(f"{role_name}: {score}%")

            # Skill Gap
            with st.expander("Skill Gap"):
                if r["missing_skills"]:
                    for skill in r["missing_skills"]:
                        st.write(f"❌ {skill}")
                else:
                    st.success("No major gaps")

            # AI Analysis
            with st.expander("AI Analysis"):
                st.write(r["gpt_analysis"])

            # Resume Highlight
            with st.expander("Resume Highlight"):
                keywords = list(r["skills"].keys())
                highlighted = highlight_text(raw_texts[r["name"]], keywords)
                st.markdown(highlighted, unsafe_allow_html=True)

            # =========================
            # 🧠 MATCHED SECTIONS (NEW)
            # =========================
            with st.expander("Matched Sections"):
                text_lines = raw_texts[r["name"]].split("\n")
                keywords = list(r["skills"].keys())

                matched = [
                    line for line in text_lines
                    if any(k.lower() in line.lower() for k in keywords)
                ]

                if matched:
                    for line in matched[:10]:
                        st.write("👉 " + line)
                else:
                    st.write("No strong matches found")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Advanced AI Resume Screening System")
