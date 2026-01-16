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
url = st.text_input("Enter URL")

if st.button("Check URL"):
    features = extract_features(url)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    if pred == 0:
        st.error(f"⚠️ Phishing URL ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Legitimate URL ({prob*100:.2f}%)")
