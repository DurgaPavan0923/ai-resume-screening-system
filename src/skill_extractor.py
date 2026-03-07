def load_skills(path="data/skills.txt"):

    with open(path, "r") as f:
        skills = [line.strip().lower() for line in f.readlines()]

    return skills


def extract_skills(text, skills_db):

    found = []

    text = text.lower()

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found
