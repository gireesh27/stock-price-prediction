# bulk_insert.py

from db import connect_to_database
from datetime import datetime
from pymongo.errors import BulkWriteError

client = connect_to_database()
db = client["stock-price-prediction"]

def insert_many_records(symbol, records):
    """
    Inserts only NEW 5-minute stock candles into MongoDB.
    Duplicate timestamps (_id) are automatically skipped.
    """

    if not isinstance(records, list) or len(records) == 0:
        print(f"[{symbol}] No records to insert")
        return

    collection = db[symbol.upper()]

    formatted_docs = []

    for rec in records:
        try:
            date_obj = datetime.strptime(rec["date"], "%Y-%m-%d %H:%M:%S")

            doc = {
                "_id": date_obj,
                "Date": date_obj,
                "Open": float(rec["open"]),
                "High": float(rec["high"]),
                "Low": float(rec["low"]),
                "Close": float(rec["close"]),
                "Volume": int(rec["volume"] or 0),  # <-- FIXED
                "Adj_Close": float(rec.get("adj_close", rec["close"])),
            }

            formatted_docs.append(doc)

        except Exception as e:
            print(f"[{symbol}] Skipped record {rec} → {e}")

    # Prevent insert_many([]) error
    if not formatted_docs:
        print(f"[{symbol}] No valid records to insert")
        return

    try:
        result = collection.insert_many(formatted_docs, ordered=False)
        print(f"[{symbol}] Inserted {len(result.inserted_ids)} new records")

    except BulkWriteError:
        print(f"[{symbol}] Duplicate timestamps skipped. Others inserted.")
