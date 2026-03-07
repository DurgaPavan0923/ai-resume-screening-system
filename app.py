import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import preprocess_text
from src.skill_extractor import load_skills, extract_skills
from src.job_predictor import predict_role
from src.pdf_parser import extract_text_from_pdf


st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Screening & Job Match System")

st.markdown("Smart resume screening using NLP and Machine Learning")

st.divider()


col1, col2 = st.columns([2,1])


with col1:

    job_description = st.text_area(
        "📄 Enter Job Description",
        height=200
    )


with col2:

    uploaded_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )


skills_db = load_skills()


def extract_keywords(text):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10)

    X = vectorizer.fit_transform([text])

    return vectorizer.get_feature_names_out()



if st.button("🚀 Analyze Resumes"):

    if job_description and uploaded_files:

        job_clean = preprocess_text(job_description)

        job_skills = extract_skills(job_clean, skills_db)

        keywords = extract_keywords(job_clean)

        st.subheader("🔑 Important Keywords")

        st.write(", ".join(keywords))

        st.subheader("🛠 Job Skills Detected")

        st.write(", ".join(job_skills))


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


        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([job_clean] + resumes)

        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()


        results = []


        for i, resume in enumerate(resumes):

            skills = extract_skills(resume, skills_db)

            skill_match = len(set(skills) & set(job_skills))

            skill_score = skill_match / max(len(job_skills), 1)

            ats_score = (similarity[i]*0.6 + skill_score*0.4) * 100

            role = predict_role(resume)

            results.append({
                "Resume": names[i],
                "Match Score": round(ats_score,2),
                "Predicted Role": role,
                "Skills Found": ", ".join(skills)
            })


        df = pd.DataFrame(results)

        df = df.sort_values(by="Match Score", ascending=False)

        st.subheader("🏆 Resume Ranking")

        st.dataframe(df, use_container_width=True)


        st.subheader("📊 Match Scores")

        for r in results:

            st.write(f"**{r['Resume']}**")

            st.progress(r["Match Score"]/100)

            st.write(f"Match Score: {r['Match Score']}%")

            st.write(f"Predicted Role: {r['Predicted Role']}")

            st.write("---")


    else:

        st.warning("Please enter job description and upload resumes")
