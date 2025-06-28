import streamlit as st
import requests
import yfinance as yf
import sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import scipy.stats as stats

st.set_page_config(layout="wide")
st.title("📊 Stock Trend & Price Forecaster")

# --- Ticker Search Helper ---
def get_tickers(name, n=5):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": name, "quotesCount": n}
    r = requests.get(url, headers=headers, params=params)
    try:
        results = r.json()["quotes"]
        return [(x["symbol"], x.get("shortname", "No Name")) for x in results]
    except:
        st.error("❌ Error: Could not fetch ticker info.")
        return []

with st.sidebar:
    st.header("⚙️ Configuration")
    company_name = st.text_input("Enter company name", "NIFTY")
    candidates = get_tickers(company_name)
    if candidates:
        ticker = st.selectbox("Select the ticker", [f"{sym} → {name}" for sym, name in candidates])
        ticker = ticker.split(" → ")[0]
    else:
        st.stop()

    time_unit = st.selectbox("Time Frame", ["Days", "Weeks", "Months", "Years", "Hours", "Minutes"])
    unit_map = {"Days": "d", "Weeks": "wk", "Months": "mo", "Years": "y", "Hours": "h", "Minutes": "m"}
    unit_count = st.number_input("How many units?", 1, 1000, 30)
    time_frame = f"{unit_count}{unit_map[time_unit]}"

    lag_days = st.slider("Lag Days (LSTM)", 1, 30, 5)
    batch_size = st.number_input("Batch Size (LSTM)", 1, 512, 32)
    epochs = st.slider("Epochs (LSTM)", 1, 200, 10)
    alpha = st.number_input("Alpha (Focal Loss)", 0.0, 1.0, 0.25)
    gamma = st.number_input("Gamma (Focal Loss)", 0.0, 10.0, 2.0)
    lr = st.number_input("Learning Rate (Boosting)", 0.0001, 1.0, 0.01)
    split_ratio = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8)
    model_num = st.number_input("Number of Boosting Models", 1, 200, 50)
    model_choice = st.radio("Model Type for Boosting", ["Linear Regression", "Decision Tree"])

if st.button("🚀 Run Prediction"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max", interval=time_frame)
    df['netgain'] = df['Close'] - df['Open']
    df['trend'] = (df['netgain'] > 0).astype(int)
    df['target'] = df['trend'].shift(-1)
    df.dropna(inplace=True)

    cols_to_scale = ['Close', 'Open', 'High', 'Low']
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    for lag in range(1, lag_days + 1):
        for feature in cols_to_scale:
            df[f'{feature}(t-{lag})'] = df[feature].shift(lag)

    for lag in range(1, lag_days + 1):
        df[f'trend(t-{lag})'] = df['trend'].shift(lag)

    df.dropna(inplace=True)
    feature_list = ['Close', 'Open', 'High', 'Low', 'trend']
    input_features = [f'{feature}(t-{lag})' for lag in range(lag_days, 0, -1) for feature in feature_list]
    input_features += feature_list

    X = df[input_features].values
    y = df['target'].values
    X = X.reshape((X.shape[0], lag_days + 1, len(feature_list)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    def focal_loss(alpha=0.5, gamma=2.0):
        def loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
            weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
            return K.mean(weight * cross_entropy)
        return loss

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lag_days+1, len(feature_list))),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=focal_loss(alpha=alpha, gamma=gamma), optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title("Loss vs Epochs")
    ax1.legend()
    st.pyplot(fig1)

    y_scores = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    y_pred = (y_scores > best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"### 📌 Best Threshold: {best_threshold:.2f}")
    st.write(f"### ✅ Accuracy: {acc:.4f}")
    st.write("### Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm))

    # --- Boosting Model ---
    df_f = stock.history(period="max", interval=time_frame)
    df_f["target"] = df_f["Close"].shift(-1)
    df_f["volat"] = df_f["High"] - df_f["Low"]
    df_f["net_gain"] = df_f["Close"] - df_f["Open"]
    df_f.dropna(inplace=True)

    test_data = df_f.iloc[int((split_ratio)*len(df_f)):].copy()
    df_f = df_f.iloc[:int((split_ratio)*len(df_f))].copy()

    m = []
    df_f["y_pred"] = np.mean(df_f["target"])

    for _ in range(model_num):
        df_f["res"] = df_f["target"] - df_f["y_pred"]
        reg_model = LinearRegression() if model_choice == "Linear Regression" else DecisionTreeRegressor()
        reg_model.fit(df_f[["Close", "Open", "volat", "net_gain"]], df_f["res"])
        df_f["y_pred"] += lr * reg_model.predict(df_f[["Close", "Open", "volat", "net_gain"]])
        m.append(reg_model)

    test_data["y_pred_test"] = np.mean(df_f["target"])
    for model in m:
        test_data["y_pred_test"] += lr * model.predict(test_data[["Close", "Open", "volat", "net_gain"]])

    mse_test = mean_squared_error(test_data["target"], test_data["y_pred_test"])
    r2_test = r2_score(test_data["target"], test_data["y_pred_test"])

    st.write(f"### 📉 Test MSE: {mse_test:.4f} | R2: {r2_test:.4f}")

    fig2, ax2 = plt.subplots()
    ax2.scatter(test_data["Open"], test_data["target"], label="True")
    ax2.scatter(test_data["Open"], test_data["y_pred_test"], label="Pred")
    ax2.set_title("True vs Predicted Close")
    ax2.legend()
    st.pyplot(fig2)

    
    fig3, ax3 = plt.subplots()
    ax3.hist(df_f["res"], bins=50, edgecolor='black')
    ax3.set_title("Error Distribution (Train)")
    st.pyplot(fig3)

    test_data["res"] = test_data["target"] - test_data["y_pred_test"]
    fig3, ax3 = plt.subplots()
    ax3.hist(test_data["res"], bins=50, edgecolor='black')
    ax3.set_title("Error Distribution (Test)")
    st.pyplot(fig3)

    x = np.mean(abs(test_data["res"]))
    st.write(f"📏 Mean Absolute Error over test data: {x:.4f}")

    data = df_f["res"].dropna().values  

    distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max',
                    't', 'laplace', 'cauchy', 'gennorm', 'skewnorm']

    results = []

    st.markdown("### 📊 Fitting Distributions to Error Data")
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * log_likelihood
            results.append((dist_name, aic, params))
        except Exception as e:
            st.write(f"{dist_name} failed: {e}")

    
    results.sort(key=lambda x: x[1])

    st.markdown("#### 📉 AIC Values for Fitted Distributions")
    for dist_name, aic, _ in results:
        st.write(f"{dist_name}: AIC = {aic:.2f}")

    best_dist_name, _, best_params = results[0]
    best_dist = getattr(stats, best_dist_name)
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = best_dist.pdf(x, *best_params)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    ax.plot(x, pdf_fitted, 'r-', label=f'{best_dist_name} fit')
    ax.legend()
    ax.set_title(f'Best PDF Fit: {best_dist_name}')
    ax.grid(True)

    st.pyplot(fig)
