from openai import OpenAI

client = OpenAI()

def analyze_resume(resume_text, job_desc):
    prompt = f"""
    Analyze this resume against the job description.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}

    Provide:
    - Match summary
    - Strengths
    - Weaknesses
    - Final recommendation (Yes/No)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
