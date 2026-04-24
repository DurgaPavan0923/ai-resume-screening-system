import streamlit as st
import pandas as pd
import plotly.express as px

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_role
from src.experience_extractor import extract_experience
from src.education_parser import extract_education
from src.explainer import generate_explanation
from src.highlighter import highlight_text

from utils.helpers import validate_input, format_skills
from config import SKILLS_PATH


# =========================
# 🎨 CSS
# =========================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;600&family=Inter:wght@400;500&display=swap');

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
        color: #010c0d !important;
    }

    /* ===== HEADINGS ===== */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #00e6ff !important;
    }

    .card {
        background: rgba(0,0,0,0.6);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# CONFIG
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
# HELPERS
# =========================
def skill_match_score(jd_skills, resume_skills, skills_db):
    if not jd_skills:
        return 0
    total = sum(skills_db.get(s, 1) for s in jd_skills)
    matched = sum(skills_db.get(s, 1) for s in resume_skills if s in jd_skills)
    return matched / total if total else 0


def get_decision(score):
    if score >= 75:
        return "<span style='color:lime'>🟢 Hire</span>"
    elif score >= 50:
        return "<span style='color:orange'>🟡 Consider</span>"
    else:
        return "<span style='color:red'>🔴 Reject</span>"


def skill_gap(jd_skills, resume_skills):
    return [s for s in jd_skills if s not in resume_skills]


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("Dashboard")
    st.write("AI Resume Analyzer")

    st.markdown("### Tips")
    st.write("✔ Use detailed job descriptions")
    st.write("✔ Add skills")
    st.write("✔ Upload multiple resumes")


# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">AI Resume Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart hiring powered by AI</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# =========================
# INPUT
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area("Job Description", height=200)

with col2:
    files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)


# =========================
# ANALYZE
# =========================
if st.button("Analyze Candidates"):

    valid, msg = validate_input(job_desc, files)
    if not valid:
        st.warning(msg)
        st.stop()

    skills_db = load_skills(SKILLS_PATH)
    job_clean = clean_text(job_desc)

    results = []
    raw_texts = {}

    for file in files:
        try:
            text = parse_pdf(file)

            if not text.strip():
                continue

            raw_texts[file.name] = text
            clean = clean_text(text)

            similarity_score = compute_similarity(job_clean, clean, vectorizer)

            skills = extract_skills(clean, skills_db) or {}
            jd_skills = extract_skills(job_clean, skills_db) or {}

            skill_score = skill_match_score(jd_skills, skills, skills_db)

            experience = extract_experience(clean)
            education = extract_education(clean)

            role = predict_role(clean, model, vectorizer)

            # GPT SAFE FALLBACK
            try:
                from src.gpt_analyzer import analyze_resume
                gpt_analysis = analyze_resume(text, job_desc)
            except:
                gpt_analysis = "AI analysis not available"

            final_score = (
                0.5 * similarity_score +
                0.3 * skill_score +
                0.2 * min(experience / 10, 1)
            )

            final_score = max(0, min(final_score, 1))
            final_score_percent = round(final_score * 100, 2)

            decision = get_decision(final_score_percent)
            missing_skills = skill_gap(jd_skills, skills)

            explanation = generate_explanation(skills, experience, role)

            results.append({
                "name": file.name,
                "score": final_score_percent,
                "skills": skills,
                "role": role,
                "match_percent": round(skill_score * 100, 2),
                "experience": experience,
                "education": education,
                "explanation": explanation,
                "gpt_analysis": gpt_analysis,
                "decision": decision,
                "missing_skills": missing_skills
            })

        except Exception as e:
            st.error(f"{file.name}: {e}")

    # =========================
    # RESULTS
    # =========================
    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        df = pd.DataFrame(results)

        # DASHBOARD
        st.subheader("Recruiter Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Score", round(df["score"].mean(), 2))
        col2.metric("Avg Experience", round(df["experience"].mean(), 1))
        col3.metric("Candidates", len(df))

        # Plotly Charts
        colA, colB = st.columns(2)

        with colA:
            fig = px.bar(df, x="name", y="score", color="score", title="Candidate Scores")
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig2 = px.scatter(df, x="experience", y="score", size="score",
                              color="score", title="Experience vs Score")
            st.plotly_chart(fig2, use_container_width=True)
            

       # =========================
        # CANDIDATES GRID
        # =========================
        st.subheader("Ranked Candidates")

        cols = st.columns(2)

        for i, r in enumerate(results):
            with cols[i % 2]:

                st.markdown(f"""
                <div class="card">
                    <h3>{r['name']}</h3>
                    <p>{r['decision']}</p>
                    <p>Score: {r['score']}%</p>
                    <p>Role: {r['role']}</p>
                    <p>Skills: {format_skills(r['skills'])}</p>
                    <p>Experience: {r['experience']} years</p>
                    <p>Education: {', '.join(r['education']) if r['education'] else "Not detected"}</p>
                    <p>{r['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(r["score"] / 100)
                
                # Skill Gap
                with st.expander("Skill Gap Analysis"):
                    if r["missing_skills"]:
                        st.write(", ".join(r["missing_skills"]))
                    else:
                        st.write("No major gaps")
                
                # AI Analysis
                with st.expander("AI Analysis"):
                    st.write(r["gpt_analysis"])

                # Resume Highlight
                with st.expander("Resume Highlight"):
                    keywords = list(r["skills"].keys()) if isinstance(r["skills"], dict) else r["skills"]
                    highlighted = highlight_text(raw_texts[r["name"]], keywords)
                    st.markdown(highlighted, unsafe_allow_html=True)

        # SUMMARY
        top = results[0]

        st.subheader("Summary")
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
st.markdown("Advanced AI Resume Screening System")
