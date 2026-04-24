from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.similarity import compute_similarity
from src.skill_extractor import extract_skills
from src.experience_extractor import extract_experience
from src.education_parser import extract_education
from src.job_predictor import predict_role
from src.explainer import generate_explanation


def process_resume(file, job_desc, model, vectorizer, skills_db):
    text = parse_pdf(file)
    clean = clean_text(text)

    similarity_score = compute_similarity(job_desc, clean, vectorizer)

    skills = extract_skills(clean, skills_db)
    jd_skills = extract_skills(job_desc, skills_db)

    experience = extract_experience(clean)
    education = extract_education(clean)

    role = predict_role(clean, model, vectorizer)

    explanation = generate_explanation(skills, experience, role)

    return {
        "name": file.name,
        "score": similarity_score,
        "skills": skills,
        "role": role,
        "experience": experience,
        "education": education,
        "explanation": explanation,
        "raw_text": text
    }
