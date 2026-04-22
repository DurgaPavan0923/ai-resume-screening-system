import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import predict_role, load_model
from config import SKILLS_PATH

# Page config
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

# ===== Custom UI Styling =====
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.title {
    font-size: 42px;
    font-weight: bold;
    color: #00adb5;
}
.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)

# Input: Job Description
job_desc = st.text_area("📄 Enter Job Description", height=150)

# Upload resumes
files = st.file_uploader("📂 Upload Resumes (PDF)", accept_multiple_files=True)

# Analyze button
if st.button("🚀 Analyze"):
    if not job_desc or not files:
        st.warning("Please provide job description and upload resumes")
    else:
        skills_db = load_skills(SKILLS_PATH)
        model, vectorizer = load_model()

        job_clean = clean_text(job_desc)
        job_vec = vectorizer.transform([job_clean])

        results = []

        for file in files:
            text = parse_pdf(file)
            clean = clean_text(text)

            vec = vectorizer.transform([clean])
            score = cosine_similarity(job_vec, vec)[0][0]

            skills = extract_skills(clean, skills_db)
            role = predict_role(clean)

            results.append({
                "name": file.name,
                "score": round(score * 100, 2),
                "skills": skills,
                "role": role
            })

        # Sort results
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        st.subheader("🏆 Ranked Candidates")

        for r in results:
            st.markdown(f"""
            <div class="card">
                <h3>{r['name']}</h3>
                <p>🎯 Score: {r['score']}%</p>
                <p>💼 Predicted Role: {r['role']}</p>
                <p>🛠 Skills: {', '.join(r['skills'])}</p>
            </div>
            """, unsafe_allow_html=True)
