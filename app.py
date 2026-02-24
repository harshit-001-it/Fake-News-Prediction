import os
import sys
import subprocess
import time
import threading
import requests
import csv
import joblib
import pandas as pd
import re
import string

def ensure_dependencies():
    required_packages = {
        'streamlit': 'streamlit',
        'joblib': 'joblib',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'requests': 'requests'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("Dependencies installed successfully! Please re-run the script.")
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)

ensure_dependencies()

import streamlit as st

# --- Path Handling ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'user_feedback.csv')

def check_and_train():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model files missing. Training models now...")
        try:
            train_script = os.path.join(BASE_DIR, "fake_news_main.py")
            subprocess.check_call([sys.executable, train_script])
            print("Training complete.")
        except Exception as e:
            print(f"Error during training: {e}")

# Trigger training check before Streamlit launches
if __name__ == '__main__':
    try:
        from streamlit.runtime import exists
        if not exists():
            check_and_train()
    except ImportError:
        check_and_train()

# --- Auto Shutdown Logic ---
def _get_active_session_count():
    try:
        from streamlit.runtime import get_instance
        mgr = get_instance()._session_mgr
        return len(mgr.list_active_sessions())
    except Exception:
        return 0

def _shutdown_monitor():
    max_wait = 90
    waited = 0
    connected = False
    while waited < max_wait:
        count = _get_active_session_count()
        if count > 0:
            connected = True
            break
        time.sleep(2)
        waited += 2
    
    if not connected:
        os._exit(0)

    while True:
        time.sleep(15)
        count = _get_active_session_count()
        if count == 0:
            os._exit(0)

if 'monitor_started' not in st.session_state:
    st.session_state['monitor_started'] = True
    threading.Thread(target=_shutdown_monitor, daemon=True).start()

# Initialize Session State
if 'input_error' not in st.session_state:
    st.session_state.input_error = False

# --- Core Tools ---
def wordopt(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

@st.cache_resource
def load_models():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
        return None, None
    except Exception:
        return None, None

def save_feedback(text, label):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['text', 'label'])
        writer.writerow([text, label])

def fetch_webzio_news(api_key, query="politics"):
    if not api_key: return []
    url = f"https://api.webz.io/newsApi?token={api_key}&format=json&q={query}&language=english&sort=relevancy"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('posts', [])[:10]
        return []
    except Exception:
        return []

# --- Custom Styling ---
st.set_page_config(page_title="🛡️ AI Fake News Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #0b1120; color: #f8fafc; }
    h1 { background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; font-weight: 800 !important; text-align: center; padding-bottom: 20px; }
    .block-container { padding-top: 2rem; max-width: 900px; }
    .stTextArea textarea { background-color: #1e293b !important; color: #f1f5f9 !important; border: 2px solid #334155 !important; border-radius: 12px; }
    .stButton > button { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important; color: white !important; border-radius: 12px !important; transition: 0.3s; width: 100%; font-weight: 600; }
    .result-card-real { background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 25px; border-radius: 16px; text-align: center; border: 1px solid #10b981; margin-top: 15px; }
    .result-card-fake { background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%); padding: 25px; border-radius: 16px; text-align: center; border: 1px solid #ef4444; margin-top: 15px; }
    .confidence-value { font-size: 2.5rem; font-weight: 800; }
    @keyframes shake { 0%, 100% { transform: translateX(0); } 10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); } 20%, 40%, 60%, 80% { transform: translateX(10px); } }
    .shake { animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both; }
    .error-border textarea { border: 2px solid #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

model, vectorizer = load_models()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛠️ AI Control Panel")
    webzio_key = st.text_input("Webz.io API Key", type="password")
    if st.button("🚀 Retrain AI Now"):
        with st.spinner("Analyzing all datasets..."):
            subprocess.check_call([sys.executable, os.path.join(BASE_DIR, "fake_news_main.py")])
            st.success("Trained!")
            st.rerun()

# --- UI Layout ---
st.markdown("<h1>🛡️ AI Fake News Detector</h1>", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Click 'Retrain' in sidebar.")
else:
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📁 Batch Analysis", "📰 Live News"])
    
    with tab1:
        st.markdown("### Test Any Article")
        container_class = "shake error-border" if st.session_state.input_error else ""
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        news_text = st.text_area("Paste news here:", height=200)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Detect News", key="btn_detect"):
            if not news_text.strip():
                st.session_state.input_error = True
                st.rerun()
            else:
                st.session_state.input_error = False
                cleaned = wordopt(news_text)
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)[0]
                
                # Confidence
                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(vec)[0][prediction] * 100
                else:
                    confidence = 99.0

                if prediction == 1:
                    st.markdown(f'<div class="result-card-real"><h2>✅ REAL NEWS</h2><div class="confidence-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f'<div class="result-card-fake"><h2>🚨 FAKE NEWS</h2><div class="confidence-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("<p style='text-align: center;'>Accurate?</p>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                if c1.button("👍 Yes"): st.success("Thanks!")
                if c2.button("👎 No"):
                    save_feedback(news_text, 1 - prediction)
                    st.info("Saved! Use 'Retrain' later.")

    with tab2:
        st.markdown("### Upload Document")
        up_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
        if st.button("Analyze File") and up_file:
            text = up_file.read().decode('utf-8') if up_file.name.endswith('.txt') else pd.read_csv(up_file).to_string()
            cleaned = wordopt(text)
            pred = model.predict(vectorizer.transform([cleaned]))[0]
            if pred == 1: st.success("✅ REAL")
            else: st.error("🚨 FAKE")
            if st.button("🚩 Incorrect?"): save_feedback(text, 1-pred)

    with tab3:
        st.markdown("### Live Feed (Webz.io)")
        if not webzio_key: st.info("Enter API key in sidebar.")
        else:
            if st.button("Fetch News"):
                st.session_state.news = fetch_webzio_news(webzio_key)
            if 'news' in st.session_state:
                for idx, post in enumerate(st.session_state.news):
                    with st.expander(post['title']):
                        st.write(post['text'][:500] + "...")
                        if st.button(f"Analyze #{idx}"):
                            p = model.predict(vectorizer.transform([wordopt(post['text'])]))[0]
                            st.write("Result: **REAL**" if p == 1 else "Result: **FAKE**")

if __name__ == '__main__':
    try:
        from streamlit.runtime import exists
        if not exists():
            subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    except ImportError:
        pass
