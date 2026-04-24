from openai import OpenAI
import os

def analyze_resume(resume_text, job_desc):
    try:
        client = OpenAI()

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
        - Final recommendation
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI analysis unavailable (API key not configured)"
