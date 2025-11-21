# bulk_insert.py

from db import connect_to_database
from datetime import datetime
from pymongo.errors import BulkWriteError

# DB Connection
client = connect_to_database()
db = client["stock-price-prediction"]

def insert_many_records(symbol, records):
    """
    Inserts only NEW 5-minute stock candles into MongoDB.
    Duplicate timestamps (_id) are automatically skipped.
    """

    if not isinstance(records, list) or len(records) == 0:
        print("⚠️ No records to insert")
        return

    collection = db[symbol.upper()]  # collection per stock (AAPL, TSLA, etc.)

    formatted_docs = []

    for rec in records:
        try:
            # Convert string → datetime
            date_obj = datetime.strptime(rec["date"], "%Y-%m-%d %H:%M:%S")

            doc = {
                "_id": date_obj,      # unique timestamp (5min candle)
                "Date": date_obj,
                "Open": float(rec["open"]),
                "High": float(rec["high"]),
                "Low": float(rec["low"]),
                "Close": float(rec["close"]),
                "Volume": int(rec["volume"]),
                "Adj_Close": float(rec.get("adj_close", rec["close"])),
            }

            formatted_docs.append(doc)

        except Exception as e:
            print(f"⚠️ Skipped a record: {e}")

    # Bulk Insert (skip duplicates)
    try:
        result = collection.insert_many(formatted_docs, ordered=False)
        print(f"✅ Inserted {len(result.inserted_ids)} new records into {symbol}")

    except BulkWriteError:
        print(f"⚠️ Duplicate timestamps skipped for {symbol}. Others inserted.")
