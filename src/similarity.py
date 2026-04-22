from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(job_text, resume_text, vectorizer):
    job_vec = vectorizer.transform([job_text])
    resume_vec = vectorizer.transform([resume_text])

    score = cosine_similarity(job_vec, resume_vec)[0][0]

    return score
