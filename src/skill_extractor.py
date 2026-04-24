def load_skills(path):
    skills = {}

    with open(path) as f:
        for line in f:
            name, weight = line.strip().split(",")
            skills[name.lower()] = int(weight)

    return skills


def extract_skills(text, skills_db):
    found = {}
    text = text.lower()

    for skill, weight in skills_db.items():
        if skill in text:
            found[skill] = weight

    return found
