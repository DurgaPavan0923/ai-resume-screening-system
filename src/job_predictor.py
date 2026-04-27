import numpy as np


# =========================
# 🧠 RULE-BASED MULTI ROLE
# =========================
def rule_based_roles(skills):
    skills = set(skills)
    roles = []

    # 🎯 AI / Data (highest priority)
    if {"machine learning", "deep learning", "nlp", "data science"} & skills:
        roles.append("Data Scientist")

    if {"tensorflow", "pytorch", "neural networks"} & skills:
        roles.append("AI Engineer")

    # 📊 Data roles
    if {"sql", "mysql", "excel", "power bi", "tableau"} & skills:
        roles.append("Data Analyst")

    # 💻 Backend
    if {"nodejs", "django", "flask", "spring", "api"} & skills:
        roles.append("Backend Developer")

    # 🎨 Frontend
    if {"html", "css", "javascript", "react"} & skills:
        roles.append("Frontend Developer")

    # ☁️ DevOps
    if {"aws", "docker", "kubernetes"} & skills:
        roles.append("DevOps Engineer")

    # 📱 Mobile
    if {"android", "kotlin"} & skills:
        roles.append("Mobile Developer")

    return roles


# =========================
# 🔮 HYBRID MULTI-ROLE PREDICTION
# =========================
def predict_roles(text, skills, model, vectorizer):
    """
    Returns:
    - combined_roles → final roles (rule + ML)
    - ml_roles → ML confidence scores
    """

    # ✅ Rule-based roles
    rule_roles = rule_based_roles(skills)

    # ✅ ML prediction
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    # Top 3 ML roles
    top_indices = np.argsort(probs)[-3:][::-1]
    ml_roles = [(classes[i], round(probs[i] * 100, 2)) for i in top_indices]

    ml_role_names = [role for role, _ in ml_roles]

    # ✅ Combine (remove duplicates)
    combined_roles = list(dict.fromkeys(rule_roles + ml_role_names))

    # Limit roles
    combined_roles = combined_roles[:4]

    return combined_roles, ml_roles