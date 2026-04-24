from fastapi import FastAPI, UploadFile
from src.pipeline import process_resume

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile, job_desc: str):
    result = process_resume(file, job_desc)
    return result
