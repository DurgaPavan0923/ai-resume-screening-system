import pickle

MODEL_PATH = "model/job_role_model.pkl"
VEC_PATH = "model/vectorizer.pkl"

def predict_role(text):

    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VEC_PATH, "rb"))

    X = vectorizer.transform([text])

    return model.predict(X)[0]
