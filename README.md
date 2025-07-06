📊 Stock Trend & Price Prediction App
This is a Streamlit-based dashboard that predicts:

Next day stock trends (up or down) using an LSTM with Focal Loss.

Next day close prices using a custom gradient boosting-like ensemble (with Linear Regression or Decision Tree base models).

It also:

Plots ROC curves, confusion matrices, and training loss.

Shows error distribution histograms.

Fits multiple probability distributions to model residual errors (AIC comparison).

🚀 Features
🔍 Ticker search using Yahoo Finance API

📈 Flexible time frame selection (days, weeks, months, etc.)

🧠 Deep learning with LSTM (custom Focal Loss for imbalanced data)

🌳 Boosting model with adjustable learning rate and number of estimators

📝 Residual error analysis with distribution fitting (AIC-based selection)

📊 Beautiful plots and metrics on Streamlit dashboard

2️⃣ Create and activate a virtual environment
✅ For Linux / Mac
bash
python3 -m venv venv
source venv/bin/activate
✅ For Windows
bash
python -m venv venv
venv\Scripts\activate
pip install -r req.txt
python -m streamlit run app.py #run this vs code terminal 
