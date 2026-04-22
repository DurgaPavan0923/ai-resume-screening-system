import streamlit as st
import os

# Core modules
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import load_model, predict_role
from src.similarity import compute_similarity
from src.train import train

# Utils
from utils.helpers import validate_input, normalize_score, format_skills

# Config
from config import SKILLS_PATH


# =========================
# 🎨 LOAD CSS
# =========================
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass


# =========================
# 🚀 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide"
)

load_css()


# =========================
# 🔥 AUTO TRAIN MODEL (FIX)
# =========================
if not os.path.exists("models/model.pkl") or not os.path.exists("models/vectorizer.pkl"):
    st.warning("⚙️ Training model for first time... Please wait ⏳")
    train()
    st.success("✅ Model trained successfully!")


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)

st.markdown("""
Analyze resumes using AI and rank candidates based on job relevance.
""")


# =========================
# 📄 INPUT SECTION
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Enter Job Description", height=180)

with col2:
    files = st.file_uploader(
        "📂 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )


# =========================
# 🚀 ANALYZE BUTTON
# =========================
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
    # 📊 DISPLAY RESULTS
    # =========================
    if not results:
        st.error("No valid resumes processed.")
    else:
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.markdown("## 🏆 Ranked Candidates")

        for i, r in enumerate(results):
            st.markdown(f"""
            <div class="card">
                <h3>{i+1}. {r['name']}</h3>
                <p>🎯 <b>Match Score:</b> {r['score']}%</p>
                <p>💼 <b>Predicted Role:</b> {r['role']}</p>
                <p>🛠 <b>Skills:</b> {format_skills(r['skills'])}</p>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # 📈 SUMMARY
        # =========================
        top = results[0]

        st.markdown("## 📊 Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Top Candidate", top["name"])
        c2.metric("Best Score", f"{top['score']}%")
        c3.metric("Predicted Role", top["role"])


# =========================
# 📌 FOOTER
# =========================
st.markdown("---")
st.markdown("💡 Built with NLP, Machine Learning & Streamlit")
