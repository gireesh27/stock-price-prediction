from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise Exception("MONGODB_URI not set")

_client = None

def connect_to_database():
    global _client
    if _client:
        return _client

    try:
        _client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Needed for Render free tier
            tlsCAFile=None,                    # optional
            serverSelectionTimeoutMS=60000,    # increase timeout
        )
        print("MongoDB Connected")
        return _client

    except Exception as e:
        print("MongoDB connection failed:", e)
        raise e
