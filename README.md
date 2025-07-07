#  Stock Trend & Price Prediction App

## 1. Authors  
This project was collaboratively developed by:  
- **Bhavesh Chamaria** [bhavesh0609](https://github.com/bhavesh0609)  
- **Kasukurthi Sujeeth** [Princesujeeth7](https://github.com/Princesujeeth7)  


## 2. Description  

This is a **Streamlit-based dashboard** that predicts:

-  **Next-day stock trends** (up or down) using an **LSTM with Focal Loss**  
-  **Next-day close prices** using a **custom gradient boosting-like ensemble** (Linear Regression or Decision Tree base models)

It also:

-  Plots **ROC curves**, **confusion matrices**, and **training loss**  
-  Shows **error distribution histograms**  
-  Fits multiple **probability distributions** to model residual errors (**AIC comparison**)


## 3. Features  

-  Ticker search using **Yahoo Finance API**  
-  Flexible **time frame selection** (days, weeks, months, etc.)  
-  **Deep learning** with LSTM + custom **Focal Loss**  
-  Boosting model with **adjustable learning rate** and **estimators**  
-  **Residual error analysis** with **AIC-based distribution fitting**  
-  Interactive plots and metrics via **Streamlit dashboard**

---

## 4.📂 Project Structure

```bash
├── tradingapp.py          
├── tradingbot.ipynb        
├── req.txt                 
└── README.md               

```
---

## 5. Setup Instructions

### 5.1 Clone the repository

```bash
git clone https://github.com/Princesujeeth7/stockmarket.git
cd stockmarket
```
### 5.2 Create and activate a virtual environment
#### For Windows
```
python -m venv venv
venv\Scripts\activate
```
### 5.3 Install dependencies
```
pip install -r req.txt
```
### 5.4 Run the Streamlit app
```
python -m streamlit run app.py
```

## And you are all set to go 🚀
