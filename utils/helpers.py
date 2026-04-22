import os


def ensure_directory(path):
    """
    Create directory if it doesn't exist
    """
    os.makedirs(path, exist_ok=True)


def read_file(file):
    """
    Safely read uploaded file content
    """
    try:
        return file.read()
    except Exception:
        return ""


def format_skills(skills_list):
    """
    Convert skills list into readable string
    """
    if not skills_list:
        return "No skills detected"
    return ", ".join(skills_list)


def normalize_score(score):
    """
    Convert similarity score (0–1) → percentage
    """
    return round(score * 100, 2)


def validate_input(job_desc, files):
    """
    Validate user input in Streamlit
    """
    if not job_desc:
        return False, "Job description is required"

    if not files:
        return False, "Please upload at least one resume"

    return True, ""


def truncate_text(text, max_length=300):
    """
    Shorten long text for UI display
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
