import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

os.makedirs("model", exist_ok=True)

data = pd.read_csv("data/job_roles.csv")

X = data["text"]
y = data["role"]

vectorizer = TfidfVectorizer()

X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()

model.fit(X_vec, y)

pickle.dump(model, open("model/job_role_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained successfully")
