import streamlit as st
import pandas as pd
import plotly.express as px
import base64

def generate_questions(skills, role):
    skill_list = list(skills.keys()) if isinstance(skills, dict) else skills

    questions = []

    for skill in skill_list[:5]:
        questions.append(f"What is your experience with {skill}?")

    questions.append(f"Explain a real project you did as a {role}.")
    questions.append("What challenges did you face and how did you solve them?")
    questions.append("How do you optimize performance in your projects?")

    return questions

from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.skill_extractor import load_skills, extract_skills
from src.similarity import compute_similarity
from src.train import train_model
from src.job_predictor import predict_roles
from src.experience_extractor import extract_experience
from src.education_parser import extract_education
from src.highlighter import highlight_text

from utils.helpers import validate_input, format_skills
from config import SKILLS_PATH


# =========================
# CSS
# =========================
def load_css():
    st.markdown("""
    <style>

    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Inter:wght@400;500;600&family=Poppins:wght@400;500&display=swap');

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ===== BACKGROUND (ANIMATED GRADIENT) ===== */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f2027);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ===== MAIN TITLE ===== */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 52px;
        font-weight: 600;
        background: linear-gradient(90deg, #00e6ff, #00ffcc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }

    /* ===== SUBTITLE ===== */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        text-align: center;
        color: #cfd8dc !important;
        margin-bottom: 20px;
    }

    /* ===== CARDS ===== */
    .card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 20px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.15);
        color: white;
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 8px 30px rgba(0,255,255,0.25);
    }

    /* ===== HEADINGS ===== */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #00e6ff !important;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 30px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
        box-shadow: 0 0 12px rgba(0,198,255,0.6);
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0,198,255,1);
    }

    /* ===== INPUT BOX ===== */
    textarea, input {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #00c6ff;
        border-radius: 10px;
    }

    /* ===== EXPANDERS ===== */
    .st-expander {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 8px;
    }

    /* ===== TEXT VISIBILITY FIX ===== */
    label, p, span, div {
        color: #e0f7fa !important;
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
    jd_set = set(jd_skills.keys() if isinstance(jd_skills, dict) else jd_skills)
    resume_set = set(resume_skills.keys() if isinstance(resume_skills, dict) else resume_skills)
    return list(jd_set - resume_set)


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
    with st.spinner("🔍 Analyzing resumes... Please wait"):
        # ⬇️ KEEP ALL YOUR EXISTING CODE HERE (no changes)

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
            file.seek(0)
            raw_texts[file.name + "_file"] = file

            clean = clean_text(text)

            similarity_score = compute_similarity(job_clean, clean, vectorizer)

            skills = extract_skills(clean, skills_db) or {}
            jd_skills = extract_skills(job_clean, skills_db) or {}

            skill_score = skill_match_score(jd_skills, skills, skills_db)

            experience = extract_experience(clean)
            education = extract_education(clean)

            roles, ml_roles = predict_roles(clean, skills, model, vectorizer)
            role_display = ", ".join(roles)

            try:
                from src.gpt_analyzer import analyze_resume
                gpt_analysis = analyze_resume(text, job_desc)
                if not gpt_analysis.strip():
                    raise Exception()
            except:
                gpt_analysis = "⚠️ AI analysis unavailable"

            final_score = (
                0.5 * similarity_score +
                0.3 * skill_score +
                0.2 * min(experience / 10, 1)
            )

            final_score_percent = round(final_score * 100, 2)

            results.append({
                "name": file.name,
                "score": final_score_percent,
                "skills": skills,
                "role": role_display,
                "ml_roles": ml_roles,
                "experience": experience,
                "education": education,
                "gpt_analysis": gpt_analysis,
                "decision": get_decision(final_score_percent),
                "missing_skills": skill_gap(jd_skills, skills)
            })

        except Exception as e:
            st.error(f"{file.name}: {e}")

    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        df = pd.DataFrame(results)

        st.subheader("Recruiter Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Score", round(df["score"].mean(), 2))
        c2.metric("Avg Experience", round(df["experience"].mean(), 1))
        c3.metric("Candidates", len(df))

        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(px.bar(df, x="name", y="score"), use_container_width=True)
        with colB:
            st.plotly_chart(px.scatter(df, x="experience", y="score"), use_container_width=True)

        st.subheader("Ranked Candidates")

        for r in results:
        
            st.markdown("---")
        
            col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
        
            # =========================
            # COLUMN 1 → FULL ANALYSIS
            # =========================
            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>{r['name']}</h3>
                    <p>{r['decision']}</p>
                    <p><b>Score:</b> {r['score']}%</p>
                    <p><b>Education:</b> {', '.join(r['education']) if r['education'] else 'Not detected'}</p>
                    <p><b>Roles:</b> {r['role']}</p>
                    <p><b>Skills:</b> {format_skills(r['skills'])}</p>
                    <p><b>Experience:</b> {r['experience']} years</p>
                </div>
                """, unsafe_allow_html=True)
        
                st.progress(r["score"] / 100)
        
            # =========================
            # COLUMN 2 → RESUME PREVIEW
            # =========================
            with col2:
                st.markdown("### Resume Preview")
        
                file_obj = raw_texts.get(r["name"] + "_file")   # ✅ MUST BE INSIDE
        
                if file_obj:
                    file_obj.seek(0)
                    import base64
                    base64_pdf = base64.b64encode(file_obj.read()).decode('utf-8')
        
                    pdf_display = f"""
                    <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" height="500px"></iframe>
                    """
        
                    st.markdown(pdf_display, unsafe_allow_html=True)
        
                else:
                    st.warning("Preview not available")

        
            # =========================
            # COLUMN 3 → INSIGHTS
            # =========================
            with col3:
        
                st.markdown("### Role Confidence")
                for role_name, score in r["ml_roles"]:
                    st.write(f"{role_name}: {score}%")
        
                st.markdown("---")
        
                st.markdown("### Skill Gap")
                if r["missing_skills"]:
                    for skill in r["missing_skills"]:
                        st.write(f"❌ {skill}")
                else:
                    st.success("No major gaps")
        
                st.markdown("---")
        
                st.markdown("### AI Analysis")
                st.write(r["gpt_analysis"])
                
                st.markdown("---")

                st.markdown("### 🎤 Interview Questions")
                questions = generate_questions(r["skills"], r["role"])
                for q in questions:
                    st.write("👉 " + q)

# =========================
# DOWNLOAD SHORTLIST
# =========================
st.markdown("### 📥 Export Candidates")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📊 Download Shortlist (CSV)",
    data=csv_data,
    file_name="shortlisted_candidates.csv",
    mime="text/csv"
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Advanced AI Resume Screening System")
