import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as components

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import preprocess_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import predict_role
from src.pdf_parser import extract_text_from_pdf

import plotly.express as px


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume ATS",
    page_icon="🤖",
    layout="wide"
)


# ---------------- 3D BACKGROUND ----------------

particles = """
<div id="particles-js"></div>
<style>
#particles-js {
position:fixed;
width:100%;
height:100%;
top:0;
left:0;
z-index:-1;
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
</style>

<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>

<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 60},
    "size": {"value": 3},
    "move": {"speed": 1},
    "line_linked": {"enable": true},
    "color": {"value": "#00ffaa"}
  }
});
</script>
"""

components.html(particles, height=0)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.title{
font-size:48px;
font-weight:700;
text-align:center;
background: linear-gradient(90deg,#00ffaa,#00ccff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation: glow 2s infinite alternate;
}

@keyframes glow{
from {text-shadow:0 0 10px #00ffaa;}
to {text-shadow:0 0 25px #00ccff;}
}

.card{
background: rgba(255,255,255,0.1);
backdrop-filter: blur(10px);
padding:20px;
border-radius:15px;
box-shadow:0px 8px 32px rgba(0,0,0,0.3);
margin-bottom:20px;
}

.skill-tag{
display:inline-block;
background:#00ffaa;
color:black;
padding:6px 12px;
border-radius:20px;
margin:4px;
font-weight:600;
}

.rank-card{
background:rgba(255,255,255,0.15);
padding:15px;
border-radius:10px;
margin-bottom:10px;
border-left:6px solid #00ffaa;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown('<div class="title">🤖 AI Resume ATS Dashboard</div>', unsafe_allow_html=True)

st.markdown(
"<p style='text-align:center'>Automated Resume Screening using NLP + Machine Learning</p>",
unsafe_allow_html=True
)

st.divider()


# ---------------- INPUT DASHBOARD ----------------

col1, col2 = st.columns([2,1])


with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    job_description = st.text_area(
        "📄 Paste Job Description",
        height=200
    )

    st.markdown('</div>', unsafe_allow_html=True)


with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["pdf","txt"],
        accept_multiple_files=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


skills_db = load_skills()


# ---------------- KEYWORD EXTRACTION ----------------

def extract_keywords(text):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10)

    X = vectorizer.fit_transform([text])

    return vectorizer.get_feature_names_out()


# ---------------- ANALYZE BUTTON ----------------

if st.button("🚀 Analyze Resumes"):

    if job_description and uploaded_files:

        # -------- AI Typing Effect --------

        typing = st.empty()

        text = "🧠 AI is analyzing resumes..."

        display = ""

        for char in text:
            display += char
            typing.markdown(display)
            time.sleep(0.03)


        # -------- AI LOADING --------

        with st.spinner("Analyzing resumes using AI..."):

            time.sleep(2)

        job_clean = preprocess_text(job_description)

        job_skills = extract_skills(job_clean, skills_db)

        keywords = extract_keywords(job_clean)


        # -------- JOB INSIGHTS --------

        st.subheader("🧾 Job Insights")

        st.markdown("**Required Skills**")

        for s in job_skills:
            st.markdown(f'<span class="skill-tag">{s}</span>', unsafe_allow_html=True)


        resumes = []
        names = []
        raw_texts = []


        for file in uploaded_files:

            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)

            else:
                text = file.read().decode("utf-8")

            raw_texts.append(text)

            clean = preprocess_text(text)

            resumes.append(clean)

            names.append(file.name)


        # -------- VECTORIZE --------

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([job_clean] + resumes)

        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()


        # -------- SCORING --------

        results = []

        for i, resume in enumerate(resumes):

            skills = extract_skills(resume, skills_db)

            skill_match = len(set(skills) & set(job_skills))

            skill_score = skill_match / max(len(job_skills),1)

            ats_score = (similarity[i]*0.6 + skill_score*0.4) * 100

            role = predict_role(resume)

            results.append({
                "Resume": names[i],
                "Score": round(ats_score,2),
                "Role": role,
                "Skills": skills,
                "Text": raw_texts[i]
            })


        results = sorted(results, key=lambda x: x["Score"], reverse=True)


        # -------- METRICS --------

        colA,colB,colC = st.columns(3)

        colA.metric("📄 Resumes Uploaded", len(uploaded_files))
        colB.metric("🛠 Skills Detected", len(job_skills))
        colC.metric("🏆 Best Match", f"{results[0]['Score']}%")


        # -------- INTERACTIVE CHART --------

        st.subheader("📊 ATS Score Comparison")

        chart_df = pd.DataFrame({
            "Resume":[r["Resume"] for r in results],
            "Score":[r["Score"] for r in results]
        })

        fig = px.bar(
            chart_df,
            x="Resume",
            y="Score",
            color="Score",
            color_continuous_scale="viridis"
        )

        st.plotly_chart(fig, use_container_width=True)


        # -------- CANDIDATE RANKING --------

        st.subheader("🏆 Candidate Ranking")

        for i,r in enumerate(results):

            st.markdown(f"""
            <div class="rank-card">
            <b>#{i+1} {r['Resume']}</b><br>
            Predicted Role: <b>{r['Role']}</b><br>
            ATS Score: <b>{r['Score']}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["Score"]/100)


            st.markdown("**Skills Found**")

            for s in r["Skills"]:
                st.markdown(f'<span class="skill-tag">{s}</span>', unsafe_allow_html=True)


            with st.expander("Preview Resume"):
                st.write(r["Text"][:1500])

            st.divider()


    else:

        st.warning("Please upload resumes and enter job description.")


# -------- FOOTER --------

st.markdown("""
<hr>
<center>
<p style='font-size:14px;color:#aaa'>
Built with ❤️ using Streamlit | AI Resume ATS System
</p>
</center>
""", unsafe_allow_html=True)
