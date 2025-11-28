from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise Exception("Please define MONGODB_URI in your environment variables")

_client = None

def connect_to_database():
    global _client

    if _client:
        return _client

    try:
        _client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True  # IMPORTANT for Render
        )
        print("MongoDB Connected")
        return _client

    except Exception as e:
        print("MongoDB connection failed:", e)
        raise e
