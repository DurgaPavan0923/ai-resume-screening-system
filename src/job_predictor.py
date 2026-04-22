def predict_role(text, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction
