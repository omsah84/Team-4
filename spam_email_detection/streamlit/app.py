import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("../model/spam_classifier_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

# Function to predict spam or ham
def predict_spam(message):
    message_transformed = vectorizer.transform([message])  # Convert text to numerical features
    prediction = model.predict(message_transformed)  # Predict using the model
    return "🛑 Spam" if prediction[0] == 1 else "✅ Ham"

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

# Sample messages for quick testing
sample_messages = {
    "Spam Example": "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Ham Example": "Hey, are we still meeting at 5 pm today?",
}

# User input with a dropdown for sample messages
selected_sample = st.selectbox("🔹 Choose a sample message (or type your own below):", ["Type your own"] + list(sample_messages.keys()))

if selected_sample != "Type your own":
    user_message = sample_messages[selected_sample]
else:
    user_message = st.text_area("✍️ Enter SMS message here:")

# Show the user input
if user_message:
    st.write(f"📝 **Your Message:** {user_message}")

# Predict button
if st.button("🔍 Check Message"):
    if user_message.strip():
        result = predict_spam(user_message)
        st.success(f"**Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter a message!")
