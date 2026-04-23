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
# 🎨 CLEAN CSS (FIXED)
# =========================
def load_css():
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&family=Inter:wght@400&display=swap');

    /* ===== BACKGROUND ===== */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: bgMove 12s ease infinite;
    }

    @keyframes bgMove {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }

    /* ===== TITLE ===== */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        color: #00e6ff;
    }

    /* ===== SUBTITLE ===== */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        color: #cfd8dc;
    }

    /* ===== HEADINGS ===== */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #00e6ff;
    }

    /* ===== CARD ===== */
    .card {
        font-family: 'Inter', sans-serif;
        background: rgba(0,0,0,0.6);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        color: white;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(90deg, #00e6ff, #0072ff);
        color: white;
        border-radius: 10px;
        height: 45px;
        font-family: 'Poppins';
    }

    /* ===== TEXT AREA ===== */
    textarea {
        background: rgba(0,0,0,0.6) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* ===== FILE UPLOADER FIX ===== */
    div[data-testid="stFileUploader"] {
        background: rgba(0,0,0,0.6);
        padding: 10px;
        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] * {
        color: white !important;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.8);
    }

    /* ===== FIX FLOAT MENU ===== */
    div[data-testid="stToolbar"] {
        background: transparent !important;
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

        results.append({
            "name": file.name,
            "score": normalize_score(score),
            "skills": skills,
            "role": role
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
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["score"] / 100)

            with st.expander("📄 Resume Preview"):
                st.write(raw_texts[r["name"]][:1000])

    else:
        st.error("No valid resumes found")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✨ Clean & Fixed UI")
