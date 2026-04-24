def generate_explanation(skills, experience, role):
    explanation = []

    if skills:
        explanation.append(f"Strong skills in {', '.join(list(skills.keys())[:5])}")

    if experience:
        explanation.append(f"{experience}+ years experience")

    if role:
        explanation.append(f"Matches role: {role}")

    return " | ".join(explanation)
