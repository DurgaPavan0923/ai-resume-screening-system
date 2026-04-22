import pickle
from config import MODEL_PATH, VECTORIZER_PATH


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_role(text):
    model, vectorizer = load_model()
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return prediction
