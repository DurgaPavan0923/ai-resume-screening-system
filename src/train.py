import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from config import MODEL_PATH, VECTORIZER_PATH


def train():
    # Load dataset
    df = pd.read_csv("data/job_roles.csv")

    # Check columns
    if "text" not in df.columns or "role" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'role' columns")

    X = df["text"]
    y = df["role"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train Model
    model = MultinomialNB()
    model.fit(X_vec, y)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save model & vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Training completed. Models saved!")


if __name__ == "__main__":
    train()
