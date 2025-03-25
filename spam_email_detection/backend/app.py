from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model and vectorizer
model = joblib.load("../model/spam_classifier_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

def predict_spam(message):
    features = vectorizer.transform([message])
    prediction = model.predict(features)
    return "🛑 Spam" if prediction[0] == 1 else "✅ Ham"

@app.route("/", methods=["GET", "POST"])
def index():
    # Handle API (React)
    if request.is_json:
        data = request.get_json()
        message = data.get("message", "")
        if not message.strip():
            return jsonify({"prediction": "⚠️ Please enter a message."})
        result = predict_spam(message)
        return jsonify({"prediction": result})
    
    # Optional HTML form support
    prediction = None
    user_message = ""
    if request.method == "POST":
        user_message = request.form.get("message", "")
        if user_message.strip():
            prediction = predict_spam(user_message)
    return render_template("index.html", prediction=prediction, user_message=user_message)

if __name__ == "__main__":
    app.run(debug=True)
