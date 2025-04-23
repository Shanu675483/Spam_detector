import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

# --------------------------------------------
# Train model with fallback and safety checks
# --------------------------------------------
def train_model(file_path, text_col, label_col, label_map=None, model_type="naive_bayes"):
    fallback = pd.DataFrame({
        "message": [
            "You've won a free prize! Call now!",
            "This is your friend calling. Let's meet later.",
            "Congratulations! You have been selected for a free vacation.",
            "Hi, just checking in. No spam here."
        ],
        "label": ["spam", "ham", "spam", "ham"]
    })

    if not os.path.exists(file_path):
        st.warning(f"⚠️ File not found: {file_path}. Using fallback data.")
        df = fallback
    else:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception as e:
            st.warning(f"⚠️ Failed to read {file_path}: {e}. Using fallback.")
            df = fallback

    if text_col not in df.columns or label_col not in df.columns:
        st.warning(f"⚠️ Columns '{text_col}' or '{label_col}' not found. Using fallback.")
        df = fallback
        text_col = "message"
        label_col = "label"

    df = df[[text_col, label_col]].dropna()
    if label_map:
        df[label_col] = df[label_col].map(label_map)

    if df.shape[0] < 2:
        st.warning(f"⚠️ Not enough data to train. Using fallback.")
        df = fallback
        text_col = "message"
        label_col = "label"

    X_train, _, y_train, _ = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB() if model_type == "naive_bayes" else LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)

    return model, vectorizer

# --------------------------------------------
# Load all models and vectorizers
# --------------------------------------------
@st.cache_resource
def load_models():
    sms_model, sms_vectorizer = train_model("spam.csv", "v2", "v1", {'ham': 0, 'spam': 1})
    call_model, call_vectorizer = train_model("fraud_call.csv", "message", "label", {'ham': 0, 'spam': 1})
    email_model, email_vectorizer = train_model("spam_ham_dataset.csv", "text", "label_num")
    return sms_model, sms_vectorizer, call_model, call_vectorizer, email_model, email_vectorizer

# Load models once
sms_model, sms_vectorizer, call_model, call_vectorizer, email_model, email_vectorizer = load_models()

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("📲 Universal Spam Detector")
st.write("Choose the type of spam detection you want to perform:")

choice = st.selectbox("Select detection type:", ["📩 SMS Spam", "📞 Call Spam", "📧 Email Spam"])

# ---------------- SMS Spam ----------------
if choice == "📩 SMS Spam":
    st.subheader("SMS Spam Detection")
    user_input = st.text_area("Enter an SMS message:")
    if st.button("Predict SMS Spam"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        elif sms_model and sms_vectorizer:
            vec = sms_vectorizer.transform([user_input])
            pred = sms_model.predict(vec)[0]
            result = "📮 Not Spam (Ham)" if pred == 0 else "🚫 Spam"
            st.success(f"Prediction: **{result}**")

# ---------------- Call Spam ----------------
elif choice == "📞 Call Spam":
    st.subheader("Call Spam Detection")
    user_input = st.text_area("Describe the call:")
    if st.button("Predict Call Spam"):
        if not user_input.strip():
            st.warning("Please enter call content.")
        elif call_model and call_vectorizer:
            vec = call_vectorizer.transform([user_input])
            prob = call_model.predict_proba(vec)[0][1]
            if prob > 0.4:
                st.error(f"🚨 Likely Spam Call ({prob*100:.1f}% confidence)")
            else:
                st.success(f"✅ Likely Safe Call ({(1 - prob)*100:.1f}% confidence)")

# ---------------- Email Spam ----------------
elif choice == "📧 Email Spam":
    st.subheader("Email Spam Detection")
    user_input = st.text_area("Paste your email text:")
    if st.button("Predict Email Spam"):
        if not user_input.strip():
            st.warning("Please enter email content.")
        elif email_model and email_vectorizer:
            vec = email_vectorizer.transform([user_input])
            pred = email_model.predict(vec)[0]
            result = "📧 Not Spam (Ham)" if pred == 0 else "🛑 Spam Email"
            st.success(f"Prediction: **{result}**")
