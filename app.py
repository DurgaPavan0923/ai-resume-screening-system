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
# 🎨 THEME-AWARE + ANIMATION CSS
# =========================
def load_css():
    st.markdown("""
    <style>

    /* ================= BACKGROUND ANIMATION ================= */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1e3c72);
        background-size: 400% 400%;
        animation: gradientMove 12s ease infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ================= LIGHT THEME FIX ================= */
    html[data-theme="light"] .stApp {
        background: linear-gradient(-45deg, #f5f7fa, #e4ecf7, #d6e4ff, #f0f5ff);
        background-size: 400% 400%;
        animation: gradientMove 12s ease infinite;
    }

    /* ================= TEXT ================= */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #00c6ff;
    }

    html[data-theme="light"] .main-title {
        color: #0072ff;
    }

    .subtitle {
        font-size: 18px;
        color: #cfd8dc;
    }

    html[data-theme="light"] .subtitle {
        color: #333;
    }

    /* ================= CARD ================= */
    .card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        transition: 0.3s;
    }

    html[data-theme="light"] .card {
        background: rgba(255,255,255,0.9);
        color: black;
    }

    .card:hover {
        transform: translateY(-5px);
    }

    /* ================= BUTTON ================= */
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 16px;
    }

    /* ================= INPUT ================= */
    textarea {
        border-radius: 10px !important;
    }

    html[data-theme="dark"] textarea {
        background: rgba(255,255,255,0.05) !important;
        color: white !important;
    }

    html[data-theme="light"] textarea {
        background: white !important;
        color: black !important;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        backdrop-filter: blur(10px);
    }

    html[data-theme="dark"] section[data-testid="stSidebar"] {
        background: rgba(20,30,40,0.85);
    }

    html[data-theme="light"] section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.9);
    }

    /* ================= HEADINGS ================= */
    h2, h3 {
        color: #00c6ff;
    }

    html[data-theme="light"] h2,
    html[data-theme="light"] h3 {
        color: #0072ff;
    }

    </style>
    """, unsafe_allow_html=True)


# =========================
# 🚀 PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Dashboard", layout="wide")
load_css()


# =========================
# 🧠 MODEL
# =========================
@st.cache_resource
def get_model():
    return train_model()

model, vectorizer = get_model()


# =========================
# 📊 SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Dashboard")
    st.write("AI-powered resume analysis")

    st.markdown("### 💡 Tips")
    st.write("- Use detailed job descriptions")
    st.write("- Add relevant skills")
    st.write("- Upload multiple resumes")


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI & NLP</div>', unsafe_allow_html=True)

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
st.markdown("✨ AI Resume Screening System | Adaptive Theme UI")
