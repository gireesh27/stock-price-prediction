# db.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise Exception("Please define MONGODB_URI in environment variables")

_client = None

def connect_to_database():
    """
    Connects to MongoDB using a global client.
    Uses TLS and allows invalid certificates for Render free tier.
    """
    global _client

    if _client:
        return _client

    try:
        _client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Required for Render free tier
            serverSelectionTimeoutMS=20000,    # 20 seconds
            connectTimeoutMS=20000             # 20 seconds
        )
        # Test connection
        _client.admin.command('ping')
        print("MongoDB Connected")
        return _client

    except Exception as e:
        print("MongoDB connection failed:", e)
        raise e
