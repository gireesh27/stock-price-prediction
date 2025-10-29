from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import joblib
import traceback
import numpy as np
import datetime 
import os
import pandas as pd
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# 🔑 API Configuration
FINNHUB_API_KEY = "d3coi31r01qmnfgekpl0d3coi31r01qmnfgekplg"
BASE_URL = "https://finnhub.io/api/v1"

# ✅ Load your trained model (no scaler)
model = joblib.load("best_stock_model.pkl")

def compute_rsi(series, period=14):
    """
    Compute Relative Strength Index (RSI) for a pandas Series of closing prices.
    Returns a Series of RSI values (0–100).
    """
    delta = series.diff()  # price change
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Calculate exponential moving averages for smoother RSI
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

# 📈 Route: Get Real-Time Stock Quote
@app.route("/api/quote")
def get_quote():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        r = requests.get(f"{BASE_URL}/quote", params={"symbol": symbol, "token": FINNHUB_API_KEY})
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        print("Quote fetch error:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET"])
def predict_price():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        # 🔹 Load model and scaler
        if not os.path.exists("best_stock_model.pkl") or not os.path.exists("scaler.pkl"):
            return jsonify({"error": "Model or scaler not found. Please retrain first."}), 500

        model = joblib.load("best_stock_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # 🔹 Fetch recent data from local CSV for context
        csv_path = f"data/{symbol}.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": f"No data file found for {symbol}. Please wait for auto-fetch."}), 500

        df = pd.read_csv(csv_path).tail(15)  # last few rows for computing indicators

        # 🔹 Get latest live quote from Finnhub
        url = f"{BASE_URL}/quote"
        params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        # Append the latest quote to compute rolling features
        new_row = {
            "Open": data.get("o", 0),
            "High": data.get("h", 0),
            "Low": data.get("l", 0),
            "Close": data.get("c", 0),
            "Volume": data.get("v", 0),
            "Adj_Close": data.get("pc", 0)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 🔹 Recreate features used during training
        df['return_1'] = df['Close'].pct_change(1)
        df['rsi_14'] = compute_rsi(df['Close'])
        df = df.dropna().tail(1)  # take the latest valid row

        X_live = df[['Open', 'High', 'Low', 'Close', 'Volume', 'return_1', 'rsi_14']].values
        X_scaled = scaler.transform(X_live)

        # 🔹 Predict using trained model
        prediction = model.predict(X_scaled)[0]

        return jsonify({
            "symbol": symbol,
            "predicted_price": round(float(prediction), 2),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print("❌ Prediction error:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
