import streamlit as st

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_role

from utils.helpers import validate_input, normalize_score, format_skills
from config import SKILLS_PATH


# =========================
# 🎨 PREMIUM CSS
# =========================
def load_css():
    st.markdown("""
    <style>

    /* Page */
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* Title */
    .main-title {
        font-size: 48px;
        font-weight: 700;
        color: #00e6e6;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #cfd8dc;
        margin-bottom: 30px;
    }

    /* Card */
    .card {
        background: #1f2a38;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
        transition: 0.3s;
    }

    .card:hover {
        transform: scale(1.02);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        height: 50px;
        font-size: 16px;
        border: none;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Section headers */
    h2, h3 {
        color: #00e6e6;
    }

    </style>
    """, unsafe_allow_html=True)


# =========================
# 🚀 PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
load_css()


# =========================
# 🔥 MODEL (CACHED)
# =========================
@st.cache_resource
def get_model():
    return train_model()

model, vectorizer = get_model()


# =========================
# 📌 SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Dashboard Info")
    st.write("Upload resumes and match them with job descriptions.")
    st.markdown("---")
    st.write("### 💡 Tips")
    st.write("- Use detailed job descriptions")
    st.write("- Upload multiple resumes")
    st.write("- Include skills in JD")


# =========================
# 🧠 HEADER
# =========================
st.markdown('<div class="main-title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI & NLP</div>', unsafe_allow_html=True)


# =========================
# 📄 INPUT SECTION
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Job Description")
    job_desc = st.text_area(
        "",
        height=200,
        placeholder="Paste job description here..."
    )

with col2:
    st.markdown("### 📂 Upload Resumes")
    files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)


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

    with st.spinner("Analyzing resumes..."):

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

        st.markdown("## 🏆 Top Candidates")

        for i, r in enumerate(results):

            st.markdown(f"""
            <div class="card">
                <h3>#{i+1} — {r['name']}</h3>
                <p>🎯 Score: <b>{r['score']}%</b></p>
                <p>💼 Role: <b>{r['role']}</b></p>
                <p>🛠 Skills: {format_skills(r['skills'])}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["score"] / 100)

        # Summary
        top = results[0]

        st.markdown("## 📊 Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Top Candidate", top["name"])
        c2.metric("🎯 Score", f"{top['score']}%")
        c3.metric("💼 Role", top["role"])

    else:
        st.error("No valid resumes found")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✨ Built with AI, NLP & Streamlit | Premium Dashboard UI")
