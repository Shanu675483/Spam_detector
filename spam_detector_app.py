import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --------------------------------------------
# STEP 1: Load data and train model (once)
# --------------------------------------------

@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Save for future use
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer

# Load model or train if not available
if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
    model, vectorizer = train_model()
else:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

# --------------------------------------------
# STEP 2: Streamlit UI
# --------------------------------------------

st.title("📩  Spam Detector")
st.write("Enter a text message and the model will predict whether it's spam or not.")

user_input = st.text_area("📨 Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        result = "📮 Not Spam (Ham)" if prediction == 0 else "🚫 Spam"
        st.success(f"Prediction: **{result}**")
