import os
import logging
import io
from datetime import datetime
import requests
import joblib
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from threading import Thread
from db import connect_to_database
from Models.bulk_insert import insert_many_records
import gridfs

from retrain_module import STOCK_LIST, retrain_from_mongo, LOOKBACK_DAYS

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = "apidojo-yahoo-finance-v1.p.rapidapi.com"
MONGO_URI = os.getenv("MONGODB_URI")

if not RAPID_API_KEY:
    raise Exception("RAPID_API_KEY missing")
if not MONGO_URI:
    raise Exception("MONGO_URI missing")

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
fs = gridfs.GridFS(db)  # GridFS for model/scaler

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)
CORS(app)

# ===================================================
# Helper: Load model + scaler from GridFS
# ===================================================
def load_model_from_gridfs():
    try:
        model_file = fs.find_one({"filename": "best_stock_model.pkl"})
        scaler_file = fs.find_one({"filename": "scaler.pkl"})
        if not model_file or not scaler_file:
            logger.warning("Model or scaler not found in GridFS")
            return None, None

        model = joblib.load(io.BytesIO(model_file.read()))
        scaler = joblib.load(io.BytesIO(scaler_file.read()))
        return model, scaler

    except Exception as e:
        logger.error("Error loading model from GridFS: %s", e)
        return None, None

# ===================================================
# API: Get latest stock prices
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
# API: Get stock detail + predicted price
# ===================================================
@app.route("/api/stocks/<symbol>", methods=["GET"])
def get_stock_detail(symbol):
    symbol = symbol.upper()
    coll = db[symbol]

    doc = coll.find_one({}, sort=[("Date", -1)])
    if not doc:
        return jsonify({"error": "Not found"}), 404

    last_price = doc["Close"]

    model, scaler = load_model_from_gridfs()
    if not model or not scaler:
        return jsonify({
            "symbol": symbol,
            "last_price": last_price,
            "predicted_price": last_price
        })

    df = pd.DataFrame([{
        "Open": doc.get("Open") or 0.0,
        "High": doc.get("High") or 0.0,
        "Low": doc.get("Low") or 0.0,
        "Close": doc.get("Close") or 0.0,
        "Volume": doc.get("Volume") or 0.0
    }])

    try:
        X = scaler.transform(df)
        predicted_price = float(model.predict(X)[0])
    except Exception as e:
        logger.error("Prediction error: %s", e)
        predicted_price = last_price

    return jsonify({
        "symbol": symbol,
        "last_price": float(last_price),
        "predicted_price": predicted_price
    }), 200

# ===================================================
# Fetch stock data from RapidAPI
# ===================================================
def fetch_from_rapidapi(symbol, range="1mo", interval="5m"):
    url = f"https://{RAPID_API_HOST}/stock/v3/get-chart"
    params = {"symbol": symbol, "range": range, "interval": interval}
    headers = {"X-RapidAPI-Key": RAPID_API_KEY, "X-RapidAPI-Host": RAPID_API_HOST}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            logger.warning(f"Rate Limit Hit → {symbol}")
            return None

        data = response.json()
        if not data.get("chart", {}).get("result"):
            logger.warning(f"No chart data for {symbol}")
            return None

        chart = data["chart"]["result"][0]
        timestamps = chart["timestamp"]
        quote_list = chart.get("indicators", {}).get("quote", [])
        if not quote_list:
            logger.warning(f"No quote data for {symbol}")
            return None

        quote = quote_list[0]
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
        logger.error(f"Fetch error {symbol}: {e}")
        return None

# ===================================================
# Fetch + Save all stocks + retrain
# ===================================================
def fetch_and_save_all_stocks():
    logger.info("Market ALWAYS OPEN — Fetching...")
    updated = []

    for symbol in STOCK_LIST:
        try:
            candles = fetch_from_rapidapi(symbol)
            if candles:
                insert_many_records(symbol, candles)
                updated.append(symbol)
                logger.info(f"Updated {symbol}")
            else:
                logger.info(f"No data for {symbol}")
        except Exception as e:
            logger.error(f"Update error {symbol}: {e}")

    if updated:
        # Trigger retrain in background immediately after insertion
        logger.info("Starting background retrain after data insert...")
        Thread(target=retrain_from_mongo, kwargs={"days": LOOKBACK_DAYS}).start()

    return {"status": "success", "market_open": True, "updated_symbols": updated}

# ===================================================
# /api/fetch-now
# ===================================================
@app.route("/api/fetch-now")
def fetch_now():
    Thread(target=fetch_and_save_all_stocks).start()
    return jsonify({
        "status": "started",
        "message": "Fetch started in background. Model retrain will start automatically after insert."
    }), 200

# ===================================================
# Run Flask server + optional retrain at startup
# ===================================================
if __name__ == "__main__":
    logger.info("Starting retrain on startup...")
    retrain_from_mongo(days=LOOKBACK_DAYS)
    logger.info("Retrain complete. Starting Flask server...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
