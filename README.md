# 🤖 AI Resume Screening & Job Match System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-NLTK-green)
![Frontend](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

An **AI-powered recruitment assistant** that automatically analyzes resumes and ranks candidates based on how well they match a given job description.

This system simulates the core functionality of a modern **Applicant Tracking System (ATS)** using **Natural Language Processing (NLP)** and **Machine Learning**.

---

# 🚀 Project Overview

Recruiters often receive **hundreds of resumes for a single job role**, making manual screening time-consuming and inefficient.

The **AI Resume Screening & Job Match System** automates this process by:

* Extracting relevant **skills from resumes**
* Predicting **suitable job roles**
* Calculating **resume-job similarity**
* Ranking candidates based on **match percentage**

This enables **data-driven hiring decisions** and significantly reduces recruitment effort.

---

# 🔥 Key Features

✅ Job Description Input

✅ Upload Multiple Resumes (**PDF / TXT**)

✅ Resume Text Preprocessing using NLP

✅ Automatic **Skill Extraction**

✅ **Keyword Detection** from Job Description

✅ **Job Role Prediction** using Naive Bayes

✅ **TF-IDF Vectorization**

✅ **Cosine Similarity Resume Matching**

✅ **ATS Score Calculation**

✅ **Candidate Ranking Dashboard**

✅ **Resume Preview for Recruiters**

✅ Interactive **Streamlit UI**

---

# 🧠 How It Works

### Step 1 — Job Description Input

The recruiter provides a **job description** containing the required skills and responsibilities.

### Step 2 — Resume Upload

Multiple resumes can be uploaded in:

* `.pdf`
* `.txt`

### Step 3 — NLP Preprocessing

The system performs:

* Tokenization
* Stopword removal
* Text normalization
* Cleaning unwanted characters

using **NLTK**.

### Step 4 — Skill Extraction

Technical skills are extracted from resumes using a predefined **skills database**.

### Step 5 — Text Vectorization

Resumes and the job description are converted into numerical vectors using **TF-IDF**.

### Step 6 — Resume Matching

Similarity is computed using:

**Cosine Similarity**

```
Resume ↔ Job Description
```

### Step 7 — ATS Score Calculation

The final score is calculated based on:

* Semantic similarity
* Skill match percentage

### Step 8 — Job Role Prediction

A **Multinomial Naive Bayes classifier** predicts the most suitable job role.

### Step 9 — Candidate Ranking

Resumes are ranked from **highest match to lowest match**.

Results are displayed in an **ATS-style dashboard**.

---

# 🏗️ System Architecture

```
Job Description
      ↓
Resume Upload (PDF / TXT)
      ↓
Text Preprocessing (NLTK)
      ↓
Skill Extraction
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Naive Bayes Job Role Prediction
      ↓
ATS Score Calculation
      ↓
Candidate Ranking Dashboard
```

---

# 🛠️ Tech Stack

| Layer                | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| NLP                  | NLTK                    |
| Machine Learning     | Scikit-learn            |
| Model                | Multinomial Naive Bayes |
| Vectorization        | TF-IDF                  |
| Similarity Metric    | Cosine Similarity       |
| PDF Parsing          | pdfminer                |
| Frontend             | Streamlit               |

---

# 📂 Project Structure

```
AI-Resume-Job-Match-System
│
├── app.py
├── requirements.txt
│
├── src
│   ├── preprocess.py
│   ├── skill_extractor.py
│   ├── job_predictor.py
│   ├── pdf_parser.py
│   └── train.py
│
├── data
│   ├── skills.txt
│   └── job_roles.csv
│
├── assets
│   └── screenshots
│       ├── home_v2.png
│       ├── upload_v2.png
│       └── results_v2.png
│
└── README.md
```

---

# 📸 Application Preview

### 🖥️ Home Screen

![Home Screen](assets/screenshots/home_v2.png)

---

### 📂 Resume Upload & Skill Detection

![Upload Screen](assets/screenshots/upload_v2.png)

---

### 🏆 Resume Ranking & Job Role Prediction

![Results Screen](assets/screenshots/results_v2.png)

---

# ▶️ Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/DurgaPavan0923/AI-Resume-Job-Match-System.git
cd AI-Resume-Job-Match-System
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the Job Role Model

```bash
python src/train.py
```

---

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

# 🎯 Real-World Applications

This project can be used in:

🏢 HR Resume Screening Systems
🎓 University Placement Cells
🚀 Startup Hiring Platforms
🤖 AI-powered Applicant Tracking Systems (ATS)
📊 Recruitment Automation Tools

---

# 🔮 Future Enhancements

* 📄 Advanced **PDF Resume Parsing**
* 🧠 **BERT-based Resume Matching**
* 🧾 **Named Entity Recognition (NER) Skill Extraction**
* 📊 Advanced **ATS Score Calculation**
* 👩‍💼 Recruiter **Analytics Dashboard**
* 🗄️ **Database Integration**
* ☁️ Cloud Deployment (Streamlit Cloud / Render)
* 🔐 **Authentication System**
* 📈 Resume **Skill Gap Analysis**

---

# 💡 Why This Project Stands Out

✔ Solves a **real-world recruitment problem**

✔ Combines **NLP + Machine Learning**

✔ Simulates a **modern ATS system**

✔ Easily extendable to **Deep Learning**

✔ Strong **AI portfolio project**

✔ Demonstrates **industry-relevant ML applications**

---

# 👨‍💻 Author

**Rajana Durga Pavan Kumar**

B.Tech – Computer Science & Engineering (AI & ML)
Institute of Technical Education and Research (ITER)
SOA University

GitHub
https://github.com/DurgaPavan0923

---

# 📜 License

This project is licensed under the **MIT License**.
