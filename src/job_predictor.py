import numpy as np


# =========================
# 🔮 PREDICT ROLE (IMPROVED)
# =========================
def predict_role(text, model, vectorizer):
    """
    Predict job role using trained ML model
    + confidence threshold
    + fallback handling
    """

    # Transform input
    vec = vectorizer.transform([text])

    # Get probabilities
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    # Get best prediction
    max_index = np.argmax(probs)
    predicted_role = classes[max_index]
    confidence = probs[max_index]

    # =========================
    # 🧠 CONFIDENCE LOGIC
    # =========================
    if confidence < 0.40:
        return "General / Other Role"

    return predicted_role


# =========================
# 🔄 LOAD MODEL (SAFE)
# =========================
def load_model():
    """
    Optional: Only use if you're loading from .pkl
    (not needed if using train_model())
    """
    import pickle

    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        print("⚠️ Model loading failed:", e)
        return None, None