import streamlit as st
import os

# Core modules
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train
from src.job_predictor import load_model, predict_role

# Utils
from utils.helpers import validate_input, normalize_score, format_skills

# Config
from config import SKILLS_PATH


# =========================
# 🎨 CUSTOM CSS (PREMIUM UI)
# =========================
def load_css():
    st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #00adb5;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #b0bec5;
        margin-bottom: 30px;
    }

    .card {
        background: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .stButton > button {
        background-color: #00adb5;
        color: white;
        border-radius: 10px;
        height: 45px;
        font-size: 16px;
        border: none;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #007b80;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# 🚀 PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

load_css()


# =========================
# 🔥 AUTO TRAIN MODEL (FIX)
# =========================
if not os.path.exists("models/model.pkl") or not os.path.exists("models/vectorizer.pkl"):
    st.warning("⚙️ Training model... Please wait")
    train()
    st.success("✅ Model ready!")


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze resumes using AI and rank candidates intelligently</div>', unsafe_allow_html=True)


# =========================
# 📄 INPUT SECTION
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Job Description")
    job_desc = st.text_area(
        "Enter Job Description",
        height=200,
        placeholder="Paste job description here..."
    )

with col2:
    st.markdown("### 📂 Upload Resumes")
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )


# =========================
# 🚀 ANALYZE BUTTON
# =========================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Candidates"):

    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    skills_db = load_skills(SKILLS_PATH)
    model, vectorizer = load_model()

    job_clean = clean_text(job_desc)
    results = []

    with st.spinner("Analyzing resumes..."):

        for file in files:
            try:
                text = parse_pdf(file)

                if not text.strip():
                    continue

                clean = clean_text(text)

                score = compute_similarity(job_clean, clean, vectorizer)
                skills = extract_skills(clean, skills_db)
                role = predict_role(clean)

                results.append({
                    "name": file.name,
                    "score": normalize_score(score),
                    "skills": skills,
                    "role": role
                })

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    # =========================
    # 📊 RESULTS
    # =========================
    if not results:
        st.error("No valid resumes processed.")
    else:
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.markdown("## 🏆 Ranked Candidates")

        for i, r in enumerate(results):
            st.markdown(f"""
            <div class="card">
                <h3>#{i+1} — {r['name']}</h3>
                <p>🎯 <b>Score:</b> {r['score']}%</p>
                <p>💼 <b>Role:</b> {r['role']}</p>
                <p>🛠 <b>Skills:</b> {format_skills(r['skills'])}</p>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(r['score'] / 100)

        # =========================
        # 📈 SUMMARY
        # =========================
        st.markdown("## 📊 Summary")

        top = results[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("🏆 Top Candidate", top["name"])
        c2.metric("🎯 Best Score", f"{top['score']}%")
        c3.metric("💼 Role", top["role"])


# =========================
# 📌 FOOTER
# =========================
st.markdown("---")
st.markdown("💡 Built with NLP, Machine Learning & Streamlit")
