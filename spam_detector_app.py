import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# ---------------------------------------------------
# Load or train both SMS and Call spam detection models
# ---------------------------------------------------

@st.cache_resource
def train_sms_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

@st.cache_resource
def train_call_model():
    df = pd.read_csv("fraud_call.csv")
    df.columns = ['message', 'label']  # Ensure consistent columns
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# Load or train models
sms_model, sms_vectorizer = train_sms_model()
call_model, call_vectorizer = train_call_model()

# ---------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------

st.title("📲 SMS & Call Spam Detector")

option = st.selectbox("Choose the type of input you want to analyze:", ["📩 SMS Message", "📞 Call Description"])

if option == "📩 SMS Message":
    st.subheader("SMS Spam Detection")
    sms_input = st.text_area("Enter an SMS message:")
    if st.button("Predict SMS Spam"):
        if sms_input.strip() == "":
            st.warning("Please enter an SMS message.")
        else:
            input_vec = sms_vectorizer.transform([sms_input])
            pred = sms_model.predict(input_vec)[0]
            result = "📮 Not Spam (Ham)" if pred == 0 else "🚫 Spam"
            st.success(f"SMS Prediction: **{result}**")

elif option == "📞 Call Description":
    st.subheader("Call Spam Detection")
    call_input = st.text_area("Describe the call content (e.g., 'You’ve won a prize'):")
    if st.button("Predict Call Spam"):
        if call_input.strip() == "":
            st.warning("Please describe the call.")
        else:
            input_vec = call_vectorizer.transform([call_input])
            prob_spam = call_model.predict_proba(input_vec)[0][1]
            if prob_spam > 0.4:
                st.error(f"🚨 Likely Spam Call ({prob_spam*100:.1f}% confidence)")
            else:
                st.success(f"✅ Likely Safe Call ({(1 - prob_spam)*100:.1f}% confidence)")
