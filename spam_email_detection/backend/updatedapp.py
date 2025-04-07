from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)
CORS(app)

# ------------------------ Load Models & Preprocessing Tools ------------------------

ml_models = joblib.load("../model/updatedmodel/spam_classifier_model.pkl")
vectorizer = joblib.load("../model/updatedmodel/vectorizer.pkl")

# Load tokenizer
with open("../model/updatedmodel/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

# Load DL models
lstm_model = load_model("../model/updatedmodel/lstm_model.h5")
conv_model = load_model("../model/updatedmodel/conv_model.h5")

# ------------------------ Voting-Based Prediction ------------------------

def predict_with_voting(input_mail):
    votes = []

    # ML Models (TF-IDF based)
    tfidf_features = vectorizer.transform([input_mail])
    for model in ml_models.values():
        pred = model.predict(tfidf_features)[0]
        votes.append(pred)

    # DL Models (tokenizer + padding)
    sequence = tokenizer.texts_to_sequences([input_mail])
    padded = pad_sequences(sequence, maxlen=100)

    lstm_pred = lstm_model.predict(padded, verbose=0)[0][0]
    conv_pred = conv_model.predict(padded, verbose=0)[0][0]

    lstm_vote = 1 if lstm_pred > 0.5 else 0
    conv_vote = 1 if conv_pred > 0.5 else 0

    votes.append(lstm_vote)
    votes.append(conv_vote)

    # Majority Voting
    spam_votes = votes.count(1)
    ham_votes = votes.count(0)
    result = "🛑 Spam" if spam_votes > ham_votes else "✅ Ham"
    return result

# ------------------------ API Routes ------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.is_json:
        data = request.get_json()
        message = data.get("message", "")
        if not message.strip():
            return jsonify({"prediction": "⚠️ Please enter a message."})
        result = predict_with_voting(message)
        return jsonify({"prediction": result})

    # Optional HTML UI
    prediction = None
    user_message = ""
    if request.method == "POST":
        user_message = request.form.get("message", "")
        if user_message.strip():
            prediction = predict_with_voting(user_message)
    return render_template("index.html", prediction=prediction, user_message=user_message)

# ------------------------ Run Flask App ------------------------

if __name__ == "__main__":
    app.run(debug=True)
