import streamlit as st

# Core modules
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_role

# Utils
from utils.helpers import validate_input, normalize_score, format_skills

# Config
from config import SKILLS_PATH


# =========================
# 🎨 CUSTOM CSS
# =========================
def load_css():
    st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #00adb5;
    }

    .subtitle {
        color: #b0bec5;
        margin-bottom: 25px;
    }

    .card {
        background: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }

    .stButton > button {
        background-color: #00adb5;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# 🚀 PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
load_css()


# =========================
# 🔥 TRAIN MODEL (NO PICKLE)
# =========================
@st.cache_resource
def get_model():
    return train_model()

model, vectorizer = get_model()


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze resumes and rank candidates using AI</div>', unsafe_allow_html=True)


# =========================
# 📄 INPUT
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Job Description", height=200)

with col2:
    files = st.file_uploader("📂 Upload Resumes", type=["pdf"], accept_multiple_files=True)


# =========================
# 🚀 BUTTON
# =========================
if st.button("🚀 Analyze Candidates"):

    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    skills_db = load_skills(SKILLS_PATH)
    job_clean = clean_text(job_desc)

    results = []

    with st.spinner("Analyzing..."):

        for file in files:
            try:
                text = parse_pdf(file)

                if not text.strip():
                    continue

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

            except Exception as e:
                st.error(f"{file.name}: {e}")

    # =========================
    # 📊 RESULTS
    # =========================
    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.subheader("🏆 Ranked Candidates")

        for i, r in enumerate(results):
            st.markdown(f"""
            <div class="card">
                <h3>#{i+1} — {r['name']}</h3>
                <p>🎯 Score: {r['score']}%</p>
                <p>💼 Role: {r['role']}</p>
                <p>🛠 Skills: {format_skills(r['skills'])}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["score"] / 100)

        top = results[0]

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
st.markdown("💡 AI Resume Screening System")
