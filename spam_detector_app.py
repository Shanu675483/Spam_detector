import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------
# Load & train SMS model
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
# Load & train Call model (corrected)
# --------------------------------------------
@st.cache_resource
def train_call_model():
    df = pd.read_csv("fraud_call.csv")  # This assumes the file has headers
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# --------------------------------------------
# Load & train Email model
# --------------------------------------------
@st.cache_resource
def train_email_model():
    df = pd.read_csv("spam_ham_dataset.csv")
    df = df[['text', 'label_num']]  # 'text' column for email, 'label_num' as 0=ham, 1=spam
    X_train, _, y_train, _ = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# --------------------------------------------
# Load all models
# --------------------------------------------
sms_model, sms_vectorizer = train_sms_model()
call_model, call_vectorizer = train_call_model()
email_model, email_vectorizer = train_email_model()

# --------------------------------------------
# Streamlit Interface
# --------------------------------------------

st.title("📲 Universal Spam Detector")
st.write("Choose the type of spam detection you want to perform:")

choice = st.selectbox("Select detection type:", ["📩 SMS Spam", "📞 Call Spam", "📧 Email Spam"])

if choice == "📩 SMS Spam":
    st.subheader("SMS Spam Detection")
    user_input = st.text_area("Enter an SMS message:")
    if st.button("Predict SMS Spam"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            vec = sms_vectorizer.transform([user_input])
            pred = sms_model.predict(vec)[0]
            result = "📮 Not Spam (Ham)" if pred == 0 else "🚫 Spam"
            st.success(f"Prediction: **{result}**")

elif choice == "📞 Call Spam":
    st.subheader("Call Spam Detection")
    user_input = st.text_area("Describe the call (e.g., 'You’ve won a prize'):")
    if st.button("Predict Call Spam"):
        if not user_input.strip():
            st.warning("Please enter call content.")
        else:
            vec = call_vectorizer.transform([user_input])
            prob = call_model.predict_proba(vec)[0][1]
            if prob > 0.4:
                st.error(f"🚨 Likely Spam Call ({prob*100:.1f}% confidence)")
            else:
                st.success(f"✅ Likely Safe Call ({(1 - prob)*100:.1f}% confidence)")

elif choice == "📧 Email Spam":
    st.subheader("Email Spam Detection")
    user_input = st.text_area("Paste your email text:")
    if st.button("Predict Email Spam"):
        if not user_input.strip():
            st.warning("Please enter email content.")
        else:
            vec = email_vectorizer.transform([user_input])
            pred = email_model.predict(vec)[0]
            result = "📧 Not Spam (Ham)" if pred == 0 else "🛑 Spam Email"
            st.success(f"Prediction: **{result}**")
