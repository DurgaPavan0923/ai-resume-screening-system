from src.pdf_parser import parse_pdf
from src.preprocess import clean_text
from src.similarity import compute_similarity
from src.gpt_analyzer import analyze_resume

def process_resume(file, job_desc):
    text = parse_pdf(file)
    clean = clean_text(text)

    score = compute_similarity(job_desc, clean)

    gpt_analysis = analyze_resume(text, job_desc)

    return {
        "score": score,
        "analysis": gpt_analysis
    }
