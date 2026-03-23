import os
import sys
import subprocess
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def ensure_dependencies():
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib',
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
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)

ensure_dependencies()

def wordopt(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

def solve_dataset_logic(file_path):
    """Detects the dataset type and returns a cleaned DataFrame with text and label."""
    try:
        fname = os.path.basename(file_path).lower()
        # Handle special CSV formats
        sep = ','
        if 'train (2)' in fname:
            sep = ';'
            
        df = pd.read_csv(file_path, on_bad_lines='skip', sep=sep, engine='python')
        
        # Identify columns
        cols = [c.lower() for c in df.columns]
        text_col = next((c for c in df.columns if 'text' in c.lower()), None)
        title_col = next((c for c in df.columns if 'title' in c.lower()), None)
        
        if not text_col and title_col:
            text_col = title_col
        elif text_col and title_col:
            df['text'] = df[text_col].fillna('') + " " + df[title_col].fillna('')
            text_col = 'text'
            
        if not text_col:
            return None
            
        # Labeling Logic
        if 'label' in cols:
            lcol = df.columns[cols.index('label')]
            # WELFake mapping: 1=Fake, 0=Real -> We want Real=1, Fake=0
            if 'welfake' in file_path.lower():
                df['label'] = df[lcol].map({0: 1, 1: 0})
            elif df[lcol].dtype == 'object':
                df['label'] = df[lcol].map({'REAL': 1, 'FAKE': 0, 'real': 1, 'fake': 0, 'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0})
            else:
                df['label'] = df[lcol]
        elif 'real' in cols:
            lcol = df.columns[cols.index('real')]
            # FakeNewsNet mapping: 1=Real, 0=Fake
            df['label'] = df[lcol]
        elif 'fake' in fname or 'fake' in os.path.dirname(file_path).lower():
            df['label'] = 0
        elif 'true' in fname or 'true' in os.path.dirname(file_path).lower():
            df['label'] = 1
        else:
            return None
            
        df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
        return df.dropna()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    print("🚀 Starting Dynamic Training Sequence...")
    
    base_data_path = r"C:\Users\harsh_2pgm3oe\OneDrive\Documents\Coding\All Docs\Fake News"
    user_feed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_feedback.csv')
    
    all_dfs = []
    
    # 1. Scan Main Datasets
    print(f"Scanning directory: {base_data_path}")
    for root, dirs, files in os.walk(base_data_path):
        for file in files:
            if file.endswith('.csv'):
                fpath = os.path.join(root, file)
                print(f" -> Found: {file}")
                df = solve_dataset_logic(fpath)
                if df is not None:
                    all_dfs.append(df)
                    print(f"    Loaded {len(df)} records.")
                    
    # 2. Scan User Feedback
    if os.path.exists(user_feed_path):
        print("Found user feedback. Integrating...")
        try:
            df_feed = pd.read_csv(user_feed_path)
            all_dfs.append(df_feed)
            print(f"    Loaded {len(df_feed)} corrected samples.")
        except Exception as e:
            print(f"Error loading feedback: {e}")

    if not all_dfs:
        print("❌ No datasets found! Please check your file paths.")
        return

    print("Concatenating and cleaning data...")
    df_master = pd.concat(all_dfs, ignore_index=True)
    
    # Normalize labels: Ensure all are int (0 or 1)
    # Handle cases where labels might be 'REAL'/'FAKE' strings or int/str 0/1
    label_map = {
        'REAL': 1, 'FAKE': 0, 'real': 1, 'fake': 0, 
        'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0,
        '1': 1, '0': 0, 1: 1, 0: 0
    }
    df_master['label'] = df_master['label'].map(label_map).fillna(df_master['label'])
    df_master['label'] = pd.to_numeric(df_master['label'], errors='coerce')
    df_master = df_master.dropna(subset=['label'])
    df_master['label'] = df_master['label'].astype(int)
    
    df_master['text'] = df_master['text'].astype(str).apply(wordopt)
    df_master = df_master[df_master['text'].str.strip() != '']
    
    # Balance & Sample
    print(f"Total processed records: {len(df_master)}")
    print(df_master['label'].value_counts())
    
    # Take a balanced sample if data is huge (up to 100k)
    sample_size = min(len(df_master), 100000)
    df_train = df_master.sample(n=sample_size, random_state=42)
    
    x = df_train['text']
    y = df_train['label']
    
    print("Splitting & Vectorizing...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=10000)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    
    # SGDClassifier with 'hinge' loss behaves like a linear SVM/PAC
    print("Training Best-Fit Model (SGD Classifier)...")
    model = SGDClassifier(loss='hinge', penalty=None, learning_rate='optimal', max_iter=1000, random_state=42)
    model.fit(xv_train, y_train)
    
    pred = model.predict(xv_test)
    acc = accuracy_score(y_test, pred)
    print(f"✅ Training Complete. Accuracy: {acc:.4f}")
    
    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(base_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(base_dir, 'tfidf_vectorizer.pkl'))
    print(f"🏆 Model & Vectorizer updated in {base_dir}")

if __name__ == "__main__":
    main()
