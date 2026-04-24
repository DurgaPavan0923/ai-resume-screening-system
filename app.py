import streamlit as st
import pandas as pd

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_role

# ✅ NEW IMPORTS
from src.experience_extractor import extract_experience
from src.education_parser import extract_education
from src.explainer import generate_explanation
from src.highlighter import highlight_text

from utils.helpers import validate_input, format_skills
from config import SKILLS_PATH


# =========================
# 🎨 CSS (same as your working)
# =========================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&family=Inter:wght@400;500&display=swap');

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        color: #00e6ff;
    }

    .subtitle {
        font-family: 'Poppins';
        color: #010c0d;
    }

    h2, h3 {
        font-family: 'Poppins';
        color: #00e6ff;
    }

    .card {
        background: rgba(0,0,0,0.6);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        color: white;
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
# HELPER (NEW)
# =========================
def skill_match_score(jd_skills, resume_skills, skills_db):
    if not jd_skills:
        return 0

    total = sum(skills_db.get(s, 1) for s in jd_skills)
    matched = sum(skills_db.get(s, 1) for s in resume_skills if s in jd_skills)

    return matched / total if total else 0


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Dashboard")
    st.write("AI Resume Analyzer")


# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# =========================
# INPUT
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Job Description", height=200)

with col2:
    files = st.file_uploader("📂 Upload Resumes", type=["pdf"], accept_multiple_files=True)


# =========================
# ANALYZE
# =========================
if st.button("🚀 Analyze Candidates"):

    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    skills_db = load_skills(SKILLS_PATH)
    job_clean = clean_text(job_desc)

    results = []
    raw_texts = {}

    for file in files:
        text = parse_pdf(file)
        raw_texts[file.name] = text

        clean = clean_text(text)

        # 🔥 CORE AI LOGIC
        similarity_score = compute_similarity(job_clean, clean, vectorizer)

        skills = extract_skills(clean, skills_db)
        jd_skills = extract_skills(job_clean, skills_db)

        skill_score = skill_match_score(jd_skills, skills, skills_db)

        experience = extract_experience(clean)
        education = extract_education(clean)

        role = predict_role(clean, model, vectorizer)

        # 🎯 FINAL SCORE
        final_score = (
            0.5 * similarity_score +
            0.3 * skill_score +
            0.2 * (experience / 10)
        )

        explanation = generate_explanation(skills, experience, role)

        results.append({
            "name": file.name,
            "score": round(final_score * 100, 2),
            "skills": skills,
            "role": role,
            "match_percent": round(skill_score * 100, 2),
            "experience": experience,
            "education": education,
            "explanation": explanation
        })

    # =========================
    # RESULTS
    # =========================
    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.subheader("📊 Candidate Comparison")
        df = pd.DataFrame(results)
        st.bar_chart(df.set_index("name")["score"])

        st.subheader("🏆 Ranked Candidates")

        for r in results:

            st.markdown(f"""
            <div class="card">
                <h3>{r['name']}</h3>
                <p>🎯 Score: {r['score']}%</p>
                <p>💼 Role: {r['role']}</p>
                <p>🛠 Skills: {format_skills(r['skills'])}</p>
                <p>🎯 Skill Match: {r['match_percent']}%</p>
                <p>📅 Experience: {r['experience']} years</p>
                <p>🎓 Education: {', '.join(r['education'])}</p>
                <p>🧠 {r['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["score"] / 100)

            # ✨ HIGHLIGHTED RESUME
            with st.expander("📄 Resume Highlight"):
                highlighted = highlight_text(
                    raw_texts[r["name"]],
                    list(r["skills"].keys())
                )
                st.markdown(highlighted, unsafe_allow_html=True)

        # SUMMARY
        top = results[0]

        st.subheader("📈 Summary")
        c1, c2, c3 = st.columns(3)

        c1.metric("Top Candidate", top["name"])
        c2.metric("Score", f"{top['score']}%")
        c3.metric("Role", top["role"])

    else:
        st.error("No valid resumes found")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✨ Advanced AI Resume Screening System")
