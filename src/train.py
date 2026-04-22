import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_model():
    # Load dataset
    df = pd.read_csv("data/job_roles.csv")

    X = df["text"]
    y = df["role"]

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train model
    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer
