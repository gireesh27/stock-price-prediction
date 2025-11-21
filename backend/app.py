from flask import Flask, jsonify
from flask_cors import CORS
import requests
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime, time
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from db import connect_to_database
import pytz
from Models.bulk_insert import insert_many_records

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = "apidojo-yahoo-finance-v1.p.rapidapi.com"

STOCK_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NFLX", "NVDA", "IBM", "ORCL"
]

if not RAPID_API_KEY:
    raise Exception("❌ RAPID_API_KEY missing in .env file")

# -------------------------
# MongoDB Connection
# -------------------------
client = connect_to_database()
db = client["stock-price-prediction"]

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)
CORS(app)


# ===================================================
# CHECK IF US MARKET (NYSE/NASDAQ) IS OPEN
# 9:30 AM – 4:00 PM US/Eastern (Mon–Fri)
# ===================================================
def is_market_open():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    # Monday=0 ... Sunday=6
    if now.weekday() >= 5:  # Saturday/Sunday
        return False

    market_open = time(hour=9, minute=30)
    market_close = time(hour=16, minute=0)

    return market_open <= now.time() <= market_close


# ===================================================
# FETCH FROM RAPID API → Yahoo Finance
# ===================================================
def fetch_from_rapidapi(symbol: str, range="1d", interval="5m"):
    url = f"https://{RAPID_API_HOST}/stock/v3/get-chart"

    params = {
        "symbol": symbol,
        "range": range,
        "interval": interval
    }

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": RAPID_API_HOST
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 429:
        print(f"⚠ Rate Limit Hit → Skipping {symbol}")
        return None

    data = response.json()

    if not data.get("chart", {}).get("result"):
        print(f"❌ No chart data for {symbol}")
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


# ===================================================
# AUTO FETCH + SAVE (runs only during market hours)
# ===================================================
def fetch_and_save_all_stocks():
    print("\n🔄 Checking US market status...")

    if not is_market_open():
        print("⛔ Market CLOSED — No updates performed.\n")
        return

    print("📈 Market is OPEN — Fetching stock updates...\n")

    for symbol in STOCK_LIST:
        try:
            candles = fetch_from_rapidapi(symbol, "1d", "5m")

            if candles:
                insert_many_records(symbol, candles)
                print(f"✅ Updated: {symbol}")
            else:
                print(f"⚠ No data returned for {symbol}")

        except Exception as e:
            print(f"❌ Error updating {symbol}: {e}")

    print("✅ Market update cycle completed.\n")


# ===================================================
# BACKGROUND SCHEDULER — Runs every 5 minutes
# ===================================================
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_and_save_all_stocks, "interval", minutes=5)
scheduler.start()


# ===================================================
# RUN APP
# ===================================================
if __name__ == "__main__":
    print("🚀 Flask backend running with automatic NYSE/NASDAQ market-hour updater...")
    fetch_and_save_all_stocks()  # Initial fetch on startup
    app.run(debug=True, use_reloader=False)
