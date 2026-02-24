@echo off
echo =======================================================
echo          Fake News Detection Setup & Execution
echo =======================================================

echo.
echo [1/3] Installing dependencies...
pip install -q pandas numpy scikit-learn streamlit joblib

echo.
echo [2/3] Checking for Model Files...
IF NOT EXIST "best_model.pkl" (
    echo Model files not found! Training models now...
    echo (This may take a few minutes depending on your system)
    python fake_news_main.py
) ELSE (
    echo Models already trained! Skipping training phase.
    echo (To retrain, delete 'best_model.pkl' and run this again)
)

echo.
echo [3/3] Launching the Fake News Detector Web Interface...
echo Starting Streamlit...
streamlit run app.py

pause
