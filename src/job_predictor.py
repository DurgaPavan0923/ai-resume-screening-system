import numpy as np


# =========================
# 🧠 RULE-BASED FALLBACK
# =========================
def rule_based_role(skills):
    skills = set(skills)

    if {"html", "css", "javascript", "react", "angular"} & skills:
        return "Frontend Developer"

    if {"nodejs", "express", "django", "flask", "spring", "api"} & skills:
        return "Backend Developer"

    if {"machine learning", "deep learning", "nlp", "data science"} & skills:
        return "Data Scientist"

    if {"tensorflow", "pytorch", "neural networks"} & skills:
        return "AI Engineer"

    if {"sql", "excel", "power bi", "tableau"} & skills:
        return "Data Analyst"

    if {"aws", "docker", "kubernetes", "devops"} & skills:
        return "DevOps Engineer"

    if {"android", "kotlin", "mobile"} & skills:
        return "Mobile Developer"

    if {"cybersecurity", "security", "ethical hacking"} & skills:
        return "Security Engineer"

    return None


# =========================
# 🔮 PREDICT ROLE (HYBRID)
# =========================
def predict_role(text, skills, model, vectorizer):
    """
    Hybrid prediction:
    1. Rule-based (priority)
    2. ML model
    3. Confidence threshold
    """

    # ✅ Step 1: Rule-based override
    rule_role = rule_based_role(skills)
    if rule_role:
        return rule_role, 100  # High confidence for rule-based

    # ✅ Step 2: ML prediction
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    max_index = np.argmax(probs)
    predicted_role = classes[max_index]
    confidence = probs[max_index]

    # ✅ Step 3: Confidence threshold (lowered)
    if confidence < 0.20:
        return "General / Other Role", round(confidence * 100, 2)

    return predicted_role, round(confidence * 100, 2)


# =========================
# 🔄 LOAD MODEL (OPTIONAL)
# =========================
def load_model():
    import pickle

    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        print("⚠️ Model loading failed:", e)
        return None, None