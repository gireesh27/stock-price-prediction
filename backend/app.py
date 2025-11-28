import os
import logging
from datetime import datetime, time

import pytz
import requests
import joblib
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from db import connect_to_database
from Models.bulk_insert import insert_many_records

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = "apidojo-yahoo-finance-v1.p.rapidapi.com"
MONGO_URI = os.getenv("MONGODB_URI")

STOCK_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NFLX", "NVDA", "IBM", "ORCL"
]

if not RAPID_API_KEY:
    raise Exception("RAPID_API_KEY missing in environment variables")
if not MONGO_URI:
    raise Exception("MONGO_URI missing in environment variables")

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------
# MongoDB Connection
# -------------------------
client = connect_to_database()
db = client["stock-price-prediction"]

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)
CORS(app)  # Allow all origins for local dev

# ===================================================
# CHECK IF US MARKET (NYSE/NASDAQ) IS OPEN
# ===================================================
def is_market_open():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    if now.weekday() >= 5:
        return False
    market_open = time(hour=9, minute=30)
    market_close = time(hour=16, minute=0)
    return market_open <= now.time() <= market_close

# ===================================================
# API: Get latest prices for all stocks
# ===================================================
@app.route("/api/stocks/latest", methods=["GET"])
def get_latest_stocks():
    result = []
    for symbol in STOCK_LIST:
        coll = db[symbol]
        doc = coll.find_one({}, sort=[("Date", -1)])
        if not doc:
            continue

        price = doc.get("Close", 0)
        prev = coll.find_one({"Date": {"$lt": doc["Date"]}}, sort=[("Date", -1)])
        if prev and prev.get("Close") is not None:
            change = round(price - prev["Close"], 4)
            percent = round((change / prev["Close"]) * 100, 4)
        else:
            change = 0
            percent = 0

        result.append({
            "symbol": symbol,
            "price": price,
            "change": change,
            "percent": percent
        })
    return jsonify(result), 200

# ===================================================
# API: Get specific stock details + predicted price
# ===================================================
@app.route("/api/stocks/<symbol>", methods=["GET"])
def get_stock_detail(symbol):
    symbol = symbol.upper()
    coll = db[symbol]

    doc = coll.find_one({}, sort=[("Date", -1)])
    if not doc or doc.get("Close") is None:
        return jsonify({"error": "Symbol not found"}), 404

    last_price = doc["Close"]

    try:
        model = joblib.load("best_stock_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except Exception:
        return jsonify({
            "symbol": symbol,
            "last_price": last_price,
            "predicted_price": last_price,
            "timestamp": doc["Date"]
        }), 200

    df = pd.DataFrame([{
        "Open": doc.get("Open") or 0.0,
        "High": doc.get("High") or 0.0,
        "Low": doc.get("Low") or 0.0,
        "Close": doc.get("Close") or 0.0,
        "Volume": doc.get("Volume") or 0.0
    }])

    try:
        X = scaler.transform(df)
        pred = model.predict(X)[0]
        predicted_price = float(pred)
    except Exception:
        predicted_price = last_price

    return jsonify({
        "symbol": symbol,
        "last_price": float(last_price),
        "predicted_price": predicted_price,
        "timestamp": doc["Date"]
    }), 200

# ===================================================
# FETCH FROM RAPID API → Yahoo Finance
# ===================================================
def fetch_from_rapidapi(symbol: str, range="1d", interval="5m"):
    url = f"https://{RAPID_API_HOST}/stock/v3/get-chart"
    params = {"symbol": symbol, "range": range, "interval": interval}
    headers = {"X-RapidAPI-Key": RAPID_API_KEY, "X-RapidAPI-Host": RAPID_API_HOST}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            logger.warning(f"Rate Limit Hit → Skipping {symbol}")
            return None
        data = response.json()
        if not data.get("chart", {}).get("result"):
            logger.warning(f"No chart data for {symbol}")
            return None

        chart = data["chart"]["result"][0]
        timestamps = chart["timestamp"]
        quote = chart["indicators"]["quote"][0]

        formatted = []
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            formatted.append({
                "date": dt,
                "open": quote["open"][i],
                "high": quote["high"][i],
                "low": quote["low"][i],
                "close": quote["close"][i],
                "volume": quote["volume"][i],
                "adj_close": quote["close"][i],
            })
        return formatted
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

# ===================================================
# AUTO FETCH + SAVE (runs only during market hours)
# ===================================================
def fetch_and_save_all_stocks():
    logger.info("Checking US market status...")

    if not is_market_open():
        logger.info("Market CLOSED — No updates performed.")
        return

    logger.info("Market OPEN — Fetching stock updates...")

    for symbol in STOCK_LIST:
        try:
            candles = fetch_from_rapidapi(symbol)
            if candles:
                insert_many_records(symbol, candles)
                logger.info(f"Updated: {symbol}")
            else:
                logger.info(f"No data returned for {symbol}")
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")

    logger.info("Market update cycle completed.")

@app.route("/api/fetch-now")
def fetch_now():
    fetch_and_save_all_stocks()
    return {"status": "success", "updated": True}

# ===================================================
# Conditional Scheduler (local dev)
# ===================================================
# ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"

# if ENABLE_SCHEDULER:
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(fetch_and_save_all_stocks, "interval", minutes=5)
#     scheduler.start()
#     logger.info("Scheduler started.")
# else:
#     logger.info("Scheduler disabled.")

# ===================================================
# RUN LOCALHOST FLASK SERVER
# ===================================================
if __name__ == "__main__":
    logger.info("Starting Flask server")

    # # Do NOT run fetch in production
    # if os.getenv("RENDER") != "true":
    #     fetch_and_save_all_stocks()

    app.run(host="0.0.0.0", port=5000)