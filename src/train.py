def load_skills(path):
    with open(path, "r") as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    return skills


def extract_skills(text, skills_list):
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))
