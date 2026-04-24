def load_skills(path):
    skills = {}

    with open(path) as f:
        for line in f:
            line = line.strip().lower()

            if not line:
                continue

            # ✅ Handle both formats
            if "," in line:
                name, weight = line.split(",")
                skills[name] = int(weight)
            else:
                skills[line] = 1  # default weight

    return skills


def extract_skills(text, skills_db):
    found = {}
    text = text.lower()

    for skill, weight in skills_db.items():
        if skill in text:
            found[skill] = weight

    return found
