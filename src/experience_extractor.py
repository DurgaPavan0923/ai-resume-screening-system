import re

def extract_experience(text):
    # Matches patterns like "3 years", "5+ years"
    matches = re.findall(r'(\d+)\+?\s+years', text.lower())

    if matches:
        return max([int(x) for x in matches])

    return 0
