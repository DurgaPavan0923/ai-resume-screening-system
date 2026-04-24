def extract_education(text):
    degrees = [
        "b.tech", "bachelor", "m.tech", "master",
        "phd", "mba", "bsc", "msc"
    ]

    found = []

    text = text.lower()

    for degree in degrees:
        if degree in text:
            found.append(degree)

    return list(set(found))
