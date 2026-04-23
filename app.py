import streamlit as st
import pandas as pd

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_role

from utils.helpers import validate_input, normalize_score, format_skills
from config import SKILLS_PATH


# =========================
# 🎨 FINAL CSS (FIXED)
# =========================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&family=Inter:wght@400;500&family=Roboto:wght@400&display=swap');
    
    /* ===== TITLE ===== */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        color: #00e6ff !important;
        text-shadow: 0px 0px 20px rgba(0,255,255,0.8);
    }

    /* ===== SUBTITLE ===== */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        color: #d1e8ff !important;
    }

    /* ===== HEADINGS ===== */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #00e6ff !important;
    }

    /* ================= FIX FILE UPLOADER ================= */

    /* DARK THEME (keep as is) */
    html[data-theme="dark"] div[data-testid="stFileUploader"] {
        background: rgba(0,0,0,0.6);
        border-radius: 10px;
        padding: 10px;
    }

    html[data-theme="dark"] div[data-testid="stFileUploader"] * {
        color: white !important;
    }

    /* LIGHT THEME FIX */
    html[data-theme="light"] div[data-testid="stFileUploader"] {
        background: #1e293b !important;   /* dark card */
        border-radius: 10px;
        padding: 10px;
    }

    html[data-theme="light"] div[data-testid="stFileUploader"] * {
        color: white !important;  /* force visible text */
    }

    /* Fix file name boxes */
    html[data-theme="light"] div[data-testid="stFileUploader"] section {
        background: #111827 !important;
        border-radius: 8px;
    }

    /* Fix icons */
    html[data-theme="light"] div[data-testid="stFileUploader"] svg {
        fill: white !important;
    }

    </style>
    """, unsafe_allow_html=True)
    

# =========================
# PAGE CONFIG
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
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Dashboard")
    st.write("AI Resume Analyzer")

    st.markdown("### 💡 Tips")
    st.write("✔ Use detailed job descriptions")
    st.write("✔ Add skills")
    st.write("✔ Upload multiple resumes")


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

        score = compute_similarity(job_clean, clean, vectorizer)
        skills = extract_skills(clean, skills_db)
        role = predict_role(clean, model, vectorizer)

        jd_skills = extract_skills(job_clean, skills_db)
        match_percent = 0
        if jd_skills:
            match_percent = len(set(skills) & set(jd_skills)) / len(jd_skills) * 100

        results.append({
            "name": file.name,
            "score": normalize_score(score),
            "skills": skills,
            "role": role,
            "match_percent": round(match_percent, 2)
        })

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
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["score"] / 100)

            st.info(f"Matched due to: {', '.join(r['skills'][:5])}")

            with st.expander("📄 Resume Preview"):
                st.write(raw_texts[r["name"]][:1000])

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
st.markdown("✨ AI Resume Screening System | Clean UI")
