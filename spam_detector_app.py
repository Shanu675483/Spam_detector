import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --------------------------------------------
# 1. Train SMS spam model from CSV
# --------------------------------------------

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

# --------------------------------------------
# 2. Train Call spam model from in-memory data
# --------------------------------------------

@st.cache_resource
def train_call_model():
    data = {
        "message": [
            "Congratulations! You’ve won a cash prize.",
            "This is the IRS. You owe money. Pay now.",
            "Your number was selected for a free cruise!",
            "Click this link to claim your car warranty.",
            "We offer instant loans. Apply now!",
            "Your KYC is expiring. Call this number now.",
            "You have an overdue bill. Pay immediately.",
            "This is a court officer. Legal action pending.",
            "Call this number to stop your account closure.",
            "You are a lucky winner! Call to claim.",
            "Hello, this is your doctor’s office.",
            "This is your bank confirming a transaction.",
            "Your package is out for delivery.",
            "Reminder: Your electricity bill is due.",
            "Appointment confirmed for tomorrow.",
            "Hello, it's mom. Can you call me back?",
            "Your child's school has an announcement.",
            "Monthly subscription confirmed.",
            "Annual medical checkup reminder.",
            "Feedback call from your insurance provider."
        ],
        "label": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # spam
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # ham
        ]
    }

    df = pd.DataFrame(data)
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# --------------------------------------------
# 3. Load both models
# --------------------------------------------

sms_model, sms_vectorizer = train_sms_model()
call_model, call_vectorizer = train_call_model()

# --------------------------------------------
# 4. Streamlit UI
# --------------------------------------------

st.title("📲 SMS & Call Spam Detector")

option = st.selectbox("Choose detection type:", ["📩 SMS Spam", "📞 Call Spam"])

if option == "📩 SMS Spam":
    st.subheader("SMS Spam Detection")
    sms_input = st.text_area("Enter SMS message:")
    if st.button("Predict SMS Spam"):
        if sms_input.strip() == "":
            st.warning("Please enter an SMS message.")
        else:
            input_vec = sms_vectorizer.transform([sms_input])
            pred = sms_model.predict(input_vec)[0]
            result = "📮 Not Spam (Ham)" if pred == 0 else "🚫 Spam"
            st.success(f"Prediction: **{result}**")

elif option == "📞 Call Spam":
    st.subheader("Call Spam Detection")
    call_input = st.text_area("Describe the call content:")
    if st.button("Predict Call Spam"):
        if call_input.strip() == "":
            st.warning("Please enter call content.")
        else:
            input_vec = call_vectorizer.transform([call_input])
            prob_spam = call_model.predict_proba(input_vec)[0][1]
            if prob_spam > 0.4:
                st.error(f"🚨 Likely Spam Call ({prob_spam*100:.1f}% confidence)")
            else:
                st.success(f"✅ Likely Safe Call ({(1 - prob_spam)*100:.1f}% confidence)")
