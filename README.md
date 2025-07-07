# 📊 Stock Trend & Price Prediction App

## 👥 Authors  
This project was collaboratively developed by:  
- **Bhavesh Chamaria** – [GitHub Profile](https://github.com/bhavesh0609)  
- **Kasukurthi Sujeeth** – [GitHub Profile](https://github.com/Princesujeeth7)  


## 📝 Description  

This is a **Streamlit-based dashboard** that predicts:

- 📈 **Next-day stock trends** (up or down) using an **LSTM with Focal Loss**  
- 💰 **Next-day close prices** using a **custom gradient boosting-like ensemble** (Linear Regression or Decision Tree base models)

It also:

- ✅ Plots **ROC curves**, **confusion matrices**, and **training loss**  
- 📊 Shows **error distribution histograms**  
- 🧠 Fits multiple **probability distributions** to model residual errors (**AIC comparison**)


## 🚀 Features  

- 🔍 Ticker search using **Yahoo Finance API**  
- 📅 Flexible **time frame selection** (days, weeks, months, etc.)  
- 🧠 **Deep learning** with LSTM + custom **Focal Loss**  
- 🌳 Boosting model with **adjustable learning rate** and **estimators**  
- 📉 **Residual error analysis** with **AIC-based distribution fitting**  
- 📊 Interactive plots and metrics via **Streamlit dashboard**


## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Princesujeeth7/stockmarket.git
cd stockmarket

python -m venv venv
venv\Scripts\activate

pip install -r req.txt

python -m streamlit run app.py
