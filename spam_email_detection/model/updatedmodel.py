import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense

# ------------------------ Load and Clean Data ------------------------
raw_mail_data = pd.read_csv('./mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data['Category'] = LabelEncoder().fit_transform(mail_data['Category'])  # spam=1, ham=0

X = mail_data['Message']
Y = mail_data['Category']

# Split for ML & DL
X_train_text, X_test_text, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ------------------------ TF-IDF for Traditional ML ------------------------
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# ------------------------ Train ML Models ------------------------
ml_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(),
    "MultinomialNB": MultinomialNB()
}
ml_trained = {}
for name, model in ml_models.items():
    model.fit(X_train_tfidf, Y_train)
    acc = accuracy_score(Y_test, model.predict(X_test_tfidf))
    ml_trained[name] = model
    print(f"{name} Accuracy: {acc:.4f}")

# ------------------------ Prepare DL Data ------------------------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)
X_train_dl, X_test_dl, Y_train_dl, Y_test_dl = train_test_split(X_pad, Y, test_size=0.2, random_state=42)

# ------------------------ LSTM Model ------------------------
lstm_model = Sequential([
    Embedding(5000, 128, input_length=100),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_dl, Y_train_dl, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
lstm_acc = lstm_model.evaluate(X_test_dl, Y_test_dl, verbose=0)[1]
print(f"LSTM Accuracy: {lstm_acc:.4f}")

# ------------------------ Conv1D + GlobalAveragePooling1D Model ------------------------
conv_model = Sequential([
    Embedding(5000, 128, input_length=100),
    Conv1D(128, 5, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_model.fit(X_train_dl, Y_train_dl, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
conv_acc = conv_model.evaluate(X_test_dl, Y_test_dl, verbose=0)[1]
print(f"Conv1D Accuracy: {conv_acc:.4f}")

# ------------------------ Voting-Based Prediction ------------------------
def predict_with_voting(input_mail):
    print(f"\n📨 Input Mail: {input_mail}")
    votes = []

    # ML Models
    input_tfidf = tfidf.transform([input_mail])
    for name, model in ml_trained.items():
        pred = model.predict(input_tfidf)[0]
        votes.append(pred)
        print(f"{name} predicts: {'Spam' if pred == 1 else 'Ham'}")

    # DL Models
    seq = tokenizer.texts_to_sequences([input_mail])
    padded = pad_sequences(seq, maxlen=100)

    lstm_pred = lstm_model.predict(padded, verbose=0)[0][0]
    conv_pred = conv_model.predict(padded, verbose=0)[0][0]

    lstm_vote = 1 if lstm_pred > 0.5 else 0
    conv_vote = 1 if conv_pred > 0.5 else 0

    print(f"LSTM predicts: {'Spam' if lstm_vote == 1 else 'Ham'}")
    print(f"Conv1D predicts: {'Spam' if conv_vote == 1 else 'Ham'}")

    votes.extend([lstm_vote, conv_vote])
    spam_votes = votes.count(1)
    ham_votes = votes.count(0)

    result = 'Spam' if spam_votes > ham_votes else 'Ham'
    print(f"\n✅ Final Prediction (Majority Voting): {result} ({spam_votes} Spam / {ham_votes} Ham)")

# ------------------------ Test Voting ------------------------
predict_with_voting("Get your free $1000 gift card now!")
predict_with_voting("Hi John, please find the project update in the attached file.")

# ------------------------ Save All Assets ------------------------
os.makedirs("updatedmodel", exist_ok=True)

joblib.dump(ml_trained, "updatedmodel/spam_classifier_model.pkl")
joblib.dump(tfidf, "updatedmodel/vectorizer.pkl")

tokenizer_json = tokenizer.to_json()
with open("updatedmodel/tokenizer.json", "w") as f:
    f.write(tokenizer_json)

lstm_model.save("updatedmodel/lstm_model.h5")
conv_model.save("updatedmodel/conv_model.h5")

print("\n📦 All models, vectorizer, and tokenizer saved in 'updatedmodel/' folder.")
