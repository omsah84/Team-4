import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the data
raw_mail_data = pd.read_csv('./mail_data.csv')

# Replace null values with an empty string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label encoding: spam = 1, ham = 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0

# Separate messages and labels
X = mail_data['Message']
Y = mail_data['Category'].astype('int')  # Convert labels to integers

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Predictions & accuracy
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)

# Save model & vectorizer for Streamlit deployment
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Test with a sample email
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted."]
input_data_features = vectorizer.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Spam mail')
else:
    print('Ham mail')
