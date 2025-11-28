
# Top 10 Stocks Price Prediction (ML + Flask + MongoDB)

This project builds a **machine learning model** to predict stock prices for the **top 10 companies** (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, IBM, ORCL) using historical market data.

It demonstrates **data preprocessing, ML modeling, API integration, and real-time stock fetching** using RapidAPI, Flask, and MongoDB.


##  Dataset Overview

* **Companies:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, IBM, ORCL
* **Data Source:** Yahoo Finance via RapidAPI or CSV files
* **Columns:**

| Column    | Description              |
| --------- | ------------------------ |
| Date      | Trading day              |
| Open      | Opening price (USD)      |
| High      | Highest price of the day |
| Low       | Lowest price of the day  |
| Close     | Closing price (USD)      |
| Adj Close | Adjusted closing price   |
| Volume    | Number of shares traded  |


##  Libraries Used

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import requests
import pytz
from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
```


## 🚀 Project Workflow

1. **Load Dataset & API Setup** – Historical data CSV or RapidAPI connection.
2. **Data Preprocessing** – Handle missing values, convert date formats, scale features.
3. **Feature Engineering** – Moving averages, daily returns, high-low spreads.
4. **Exploratory Data Analysis (EDA)** –

   * Price trends
   * Correlation heatmaps
   * Volume analysis
5. **Machine Learning Model** – Train XGBoost regression or other models per stock.
6. **Model Evaluation** – MSE, RMSE, R², visual comparison of predicted vs actual prices.
7. **API Deployment** – Flask app provides endpoints:

   * `/api/stocks/latest` → Get latest stock prices with change %
   * `/api/stocks/<symbol>` → Get specific stock details + predicted price
   * `/api/fetch-now` → Fetch real-time prices from RapidAPI and update MongoDB
8. **Database Integration** – MongoDB stores historical & fetched stock data.

---

## 🧠 Machine Learning Models

| Model            | Purpose                         |
| ---------------- | ------------------------------- |
| XGBRegressor     | Predict continuous stock prices |
| StandardScaler   | Feature scaling for ML models   |
| Train/Test Split | Model evaluation and validation |

---

##  Visualizations

* Historical closing price trends per stock
* Moving averages and volatility
* Predicted vs actual price plots
* Correlation heatmaps

---

##  Folder Structure

```
stock-price-prediction
├── app.py                 # Flask API server
├── db.py                  # MongoDB connection
├── Models/
│   └── bulk_insert.py     # Helper to insert records
├── requirements.txt
├── tesla_stock.csv        # Example historical data
├── best_stock_model.pkl   # Trained ML model
├── scaler.pkl             # Scaler for preprocessing
└── README.md
```

---

##  How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/gireesh27/stock-price-prediction.git
cd stock-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables in `.env`:

```
MONGODB_URI=your_mongodb_uri
RAPID_API_KEY=your_rapidapi_key
```

4. Run the Flask server:

```bash
python app.py
```

5. Access API endpoints:

   * Latest prices → `http://127.0.0.1:5000/api/stocks/latest`
   * Specific stock → `http://127.0.0.1:5000/api/stocks/AAPL`
   * Fetch real-time → `http://127.0.0.1:5000/api/fetch-now`

---

##  Requirements

```text
flask
flask-cors
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
pytz
requests
python-dotenv
pymongo
apscheduler
```

---

##  Future Enhancements

* Add LSTM/GRU for sequence-based stock prediction
* Deploy interactive dashboard using Streamlit or Plotly
* Add real-time WebSocket updates for stock prices
* Hyperparameter tuning for ML models using GridSearchCV
* Scale to larger stock universe beyond top 10

---

## 👤 Author

**Gireesh Kasa**
B.Tech, NIT Warangal
📧 [kasagireesh@gmail.com](mailto:kasagireesh@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/gireesh-kasa-33a546250/) | [GitHub](https://github.com/gireesh27)

---

> “Stock prediction is not about certainty — it’s about probability, pattern, and precision.”
