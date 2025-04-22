import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --------------------------------------------
# STEP 1: Train and Save Model (SMS-based)
# --------------------------------------------

@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split and vectorize
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Save artifacts
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer

# Load or train
if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
    model, vectorizer = train_model()
else:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

# --------------------------------------------
# STEP 2: Streamlit Interface
# --------------------------------------------

st.title("📞 SMS & Call Spam Detection")
st.write("This app detects whether a message or call is spam.")

# Option selector
option = st.selectbox("Choose type of detection:", ["SMS Spam Detection", "Call Spam Detection"])

if option == "SMS Spam Detection":
    st.subheader("📩 SMS Message Input")
    sms_input = st.text_area("Enter SMS message here:")
    
    if st.button("Detect SMS Spam"):
        if sms_input.strip() == "":
            st.warning("Please enter an SMS message.")
        else:
            input_vec = vectorizer.transform([sms_input])
            prediction = model.predict(input_vec)[0]
            result = "📮 Not Spam (Ham)" if prediction == 0 else "🚫 Spam"
            st.success(f"SMS Prediction: **{result}**")

elif option == "Call Spam Detection":
    st.subheader("📞 Call Info Input")
    call_reason = st.text_input("Why is the caller calling you?")
    st.caption("Example: 'You won a lottery', 'Insurance policy renewal', 'Bank verification', etc.")

    if st.button("Detect Call Spam"):
        if call_reason.strip() == "":
            st.warning("Please describe the call reason.")
        else:
            input_vec = vectorizer.transform([call_reason])
            prediction = model.predict(input_vec)[0]
            result = "✅ Likely Safe Call" if prediction == 0 else "🚨 Likely Spam Call"
            st.success(f"Call Prediction: **{result}**")
