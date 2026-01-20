import streamlit as st
import joblib
import re
import ipaddress
import numpy as np
from scipy.sparse import hstack

# Load model & vectorizer
import os

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_phishing_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer_lr.pkl"))


st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🔐 Phishing URL Detection")

# ---------- HELPER FUNCTIONS ----------
def clean_url(url):
    url = url.lower()
    url = re.sub(r'https?:\/\/', '', url)
    url = re.sub(r'www\d*\.', '', url)
    url = re.sub(r'[^a-z0-9\-\.\/]', ' ', url)
    url = re.sub(r'\s+', ' ', url)
    return url.strip()

def get_url_length(url):
    return len(url)

def count_dots(url):
    return url.count('.')

def has_ip_address(url):
    try:
        ipaddress.ip_address(url)
        return 1
    except:
        return 0

def count_slashes(url):
    return url.count('/')

# ---------- FEATURE EXTRACTION ----------
def extract_features(url):
    cleaned = clean_url(url)
    text_features = vectorizer.transform([cleaned])

    manual_features = np.array([[
        get_url_length(url),
        count_dots(url),
        has_ip_address(url),
        count_slashes(url)
    ]])

    return hstack([text_features, manual_features])

# ---------- STREAMLIT UI ----------
st.set_page_config(
    page_title="PhisGuard | Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>🛡️ PhisGuard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>AI-powered Phishing URL Detection System</p>",
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.title("ℹ️ About This Project")
st.sidebar.markdown("""
**PhisGuard** detects phishing URLs using:
- TF-IDF text analysis
- Structural URL features
- Logistic Regression classifier

**Tech Stack**
- Python
- Scikit-learn
- Streamlit

📌 Deployed on Streamlit Community Cloud
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Created by Aman**")

# ---------- EXAMPLE URL BUTTONS ----------
st.markdown("### 🔗 Try Example URLs")

example_urls = {
    "✅ Legit (Google)": "https://www.google.com",
    "⚠️ Phishing (Fake Login)": "http://login-verification-update.com",
    "⚠️ Phishing (Bank Alert)": "http://bank-security-alert.co",
    "✅ Legit (GitHub)": "https://github.com"
}

col1, col2 = st.columns(2)
with col1:
    if st.button("Google"):
        st.session_state.url = example_urls["✅ Legit (Google)"]
    if st.button("Fake Login"):
        st.session_state.url = example_urls["⚠️ Phishing (Fake Login)"]

with col2:
    if st.button("GitHub"):
        st.session_state.url = example_urls["✅ Legit (GitHub)"]
    if st.button("Bank Alert"):
        st.session_state.url = example_urls["⚠️ Phishing (Bank Alert)"]

# ---------- INPUT ----------
url = st.text_input(
    "Enter a URL to analyze",
    value=st.session_state.get("url", ""),
    placeholder="https://example.com"
)

# ---------- PREDICTION ----------
if st.button("🚀 Analyze URL"):
    if not url:
        st.warning("Please enter a URL")
    else:
        features = extract_features(url)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        confidence = prob[pred] * 100

        st.markdown("---")
        st.markdown("### 🧪 Analysis Result")

        if pred == 0:
            st.error(f"⚠️ **Phishing URL Detected**")
        else:
            st.success(f"✅ **Legitimate URL**")

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.progress(int(confidence))
