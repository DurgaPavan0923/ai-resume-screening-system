import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import preprocess_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import predict_role
from src.pdf_parser import extract_text_from_pdf


# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="AI Resume ATS",
    page_icon="🤖",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
color:#2ecc71;
}

.card{
background-color:#f8f9fa;
padding:20px;
border-radius:10px;
box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

.rank-card{
background-color:#ffffff;
padding:15px;
border-radius:10px;
margin-bottom:10px;
border-left:6px solid #2ecc71;
}

.skill-box{
background:#ecf0f1;
padding:8px 12px;
border-radius:20px;
display:inline-block;
margin:4px;
}

</style>
""", unsafe_allow_html=True)


# ------------------ HEADER ------------------

st.markdown('<div class="main-title">🤖 AI Resume ATS Dashboard</div>', unsafe_allow_html=True)

st.write("Automated Resume Screening using NLP + Machine Learning")

st.divider()


# ------------------ INPUT SECTION ------------------

col1, col2 = st.columns([2,1])

with col1:

    job_description = st.text_area(
        "📄 Paste Job Description",
        height=200,
        placeholder="Paste the job description here..."
    )

with col2:

    uploaded_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["pdf","txt"],
        accept_multiple_files=True
    )


skills_db = load_skills()


# ------------------ KEYWORD EXTRACTION ------------------

def extract_keywords(text):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=12)

    X = vectorizer.fit_transform([text])

    return vectorizer.get_feature_names_out()


# ------------------ ANALYSIS BUTTON ------------------

if st.button("🚀 Analyze Resumes"):

    if job_description and uploaded_files:

        job_clean = preprocess_text(job_description)

        job_skills = extract_skills(job_clean, skills_db)

        keywords = extract_keywords(job_clean)


        # ------------------ JOB INFO ------------------

        st.subheader("🧾 Job Insights")

        colA, colB = st.columns(2)

        with colA:

            st.markdown("**🔑 Important Keywords**")

            for k in keywords:

                st.markdown(f'<span class="skill-box">{k}</span>', unsafe_allow_html=True)


        with colB:

            st.markdown("**🛠 Required Skills**")

            for s in job_skills:

                st.markdown(f'<span class="skill-box">{s}</span>', unsafe_allow_html=True)


        st.divider()


        # ------------------ READ RESUMES ------------------

        resumes = []
        names = []
        raw_text = []

        for file in uploaded_files:

            if file.name.endswith(".pdf"):

                text = extract_text_from_pdf(file)

            else:

                text = file.read().decode("utf-8")

            raw_text.append(text)

            clean = preprocess_text(text)

            resumes.append(clean)

            names.append(file.name)


        # ------------------ VECTORIZE ------------------

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([job_clean] + resumes)

        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()


        # ------------------ SCORING ------------------

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
                "Text": raw_text[i]
            })


        results = sorted(results, key=lambda x: x["Score"], reverse=True)


        # ------------------ LEADERBOARD ------------------

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

                st.markdown(f'<span class="skill-box">{s}</span>', unsafe_allow_html=True)


            with st.expander("Preview Resume"):

                st.write(r["Text"][:1500])

            st.divider()


        # ------------------ TABLE VIEW ------------------

        st.subheader("📊 Ranking Table")

        table = pd.DataFrame([
            {
                "Resume": r["Resume"],
                "ATS Score": r["Score"],
                "Predicted Role": r["Role"]
            } for r in results
        ])

        st.dataframe(table, use_container_width=True)


    else:

        st.warning("Please upload resumes and enter job description.")
