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
# 🎨 PREMIUM UI
# =========================
def load_css():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #00e6e6;
    }

    .card {
        background: #1f2a38;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
    }

    .stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)


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
    st.markdown("### 💡 Tips")
    st.write("- Use detailed job descriptions")
    st.write("- Include skills")
    st.write("- Upload multiple resumes")

    st.markdown("---")
    st.write("🚀 Built with AI & NLP")


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.write("Smart hiring powered by AI")


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

    with st.spinner("Analyzing resumes..."):

        for file in files:
            text = parse_pdf(file)
            raw_texts[file.name] = text

            clean = clean_text(text)

            score = compute_similarity(job_clean, clean, vectorizer)
            skills = extract_skills(clean, skills_db)
            role = predict_role(clean, model, vectorizer)

            # Skill match %
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

        # =========================
        # 📊 GRAPH
        # =========================
        st.subheader("📊 Candidate Comparison")

        df = pd.DataFrame(results)
        st.bar_chart(df.set_index("name")["score"])

        # =========================
        # 🏆 RESULTS
        # =========================
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

            # =========================
            # 🧠 WHY MATCHED
            # =========================
            st.info(
                f"Matched because of skills: {', '.join(r['skills'][:5])}"
            )

            # =========================
            # 📄 RESUME PREVIEW
            # =========================
            with st.expander("📄 View Resume Text"):
                st.write(raw_texts[r["name"]][:1000])

        # =========================
        # 📈 SUMMARY
        # =========================
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
st.markdown("✨ AI Resume Screening System | Next-Level Dashboard")
