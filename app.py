import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Core modules
from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import load_model, predict_role
from src.similarity import compute_similarity

# Utils
from utils.helpers import (
    validate_input,
    normalize_score,
    format_skills
)

# Config
from config import SKILLS_PATH


# =========================
# 🎨 LOAD CUSTOM CSS
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
# 🧠 HEADER
# =========================
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)

st.markdown("""
Analyze resumes instantly using AI.  
Upload resumes, match with job description, and rank candidates automatically.
""")


# =========================
# 📄 INPUT SECTION
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("📄 Enter Job Description", height=180)

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
if st.button("🚀 Analyze Candidates"):

    # Validate input
    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    # Load resources
    skills_db = load_skills(SKILLS_PATH)
    model, vectorizer = load_model()

    # Process job description
    job_clean = clean_text(job_desc)

    results = []

    # =========================
    # 🔍 PROCESS EACH RESUME
    # =========================
    with st.spinner("Analyzing resumes..."):
        for file in files:
            try:
                # Extract text from PDF
                text = parse_pdf(file)

                if not text.strip():
                    continue

                # Clean text
                clean = clean_text(text)

                # Similarity score
                score = compute_similarity(job_clean, clean, vectorizer)

                # Extract skills
                skills = extract_skills(clean, skills_db)

                # Predict role
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
        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.markdown("## 🏆 Ranked Candidates")

        # Progress bar style display
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
        st.markdown("## 📊 Summary")

        top_candidate = results[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Top Candidate", top_candidate["name"])
        col2.metric("Best Score", f"{top_candidate['score']}%")
        col3.metric("Predicted Role", top_candidate["role"])


# =========================
# 📌 FOOTER
# =========================
st.markdown("---")
st.markdown(
    "💡 Built with NLP, Machine Learning & Streamlit | AI Resume Screening System"
)
