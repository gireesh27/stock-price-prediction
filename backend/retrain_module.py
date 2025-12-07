# %%
# Single notebook cell: Load from MongoDB (last 5 days), feature engineering, tuning, stacking, save, optional auto-retrain
import os
import time
import warnings
from datetime import datetime, timedelta
import gridfs
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# %%
# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "stock-price-prediction")

STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
              "META", "NFLX", "NVDA", "IBM", "ORCL"]

LOOKBACK_DAYS = 30  # last 30 days

# %%
def connect_mongo(uri):
    if not uri:
        raise RuntimeError("MONGODB_URI not provided in environment")
    client = MongoClient(uri)
    return client

client = connect_mongo(MONGODB_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# %%
# ------------------------------------------------------
# Functions for saving/loading model in GridFS
# ------------------------------------------------------
def save_model_to_gridfs(model, scaler):
    import io

    # delete old ones
    for f in fs.find({"filename": "best_stock_model.pkl"}):
        fs.delete(f._id)
    for f in fs.find({"filename": "scaler.pkl"}):
        fs.delete(f._id)

    # Save model
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)
    fs.put(model_bytes.read(), filename="best_stock_model.pkl")

    # Save scaler
    scaler_bytes = io.BytesIO()
    joblib.dump(scaler, scaler_bytes)
    scaler_bytes.seek(0)
    fs.put(scaler_bytes.read(), filename="scaler.pkl")

    print("\n✔ Model + Scaler saved to MongoDB GridFS")

def load_model_from_gridfs():
    import io

    model_file = fs.find_one({"filename": "best_stock_model.pkl"})
    scaler_file = fs.find_one({"filename": "scaler.pkl"})

    if not model_file or not scaler_file:
        raise RuntimeError("Model/Scaler not found in GridFS")

    model = joblib.load(io.BytesIO(model_file.read()))
    scaler = joblib.load(io.BytesIO(scaler_file.read()))

    return model, scaler

# ------------------------------------------------------
# RSI
# ------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss.replace(0, 1e-8))
    return 100 - (100 / (1 + rs))

# ------------------------------------------------------
# Load last N days from MongoDB
# ------------------------------------------------------
def load_last_n_days_from_mongo(client, db_name, symbols, days=30):
    db = client[db_name]
    now_utc = datetime.utcnow()
    start = now_utc - timedelta(days=days)

    frames = []
    for sym in symbols:
        coll = db[sym.upper()]
        cursor = coll.find({"Date": {"$gte": start}})
        df = pd.DataFrame(list(cursor))
        if df.empty:
            print(f"No recent data for {sym}")
            continue

        if "_id" in df.columns and "Date" not in df.columns:
            df["Date"] = df["_id"]

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["symbol"] = sym.upper()

        expected = ["Date","Open","High","Low","Close","Volume","Adj_Close","symbol"]
        for col in expected:
            if col not in df:
                df[col] = np.nan

        frames.append(df[expected])

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames).sort_values(["symbol","Date"]).reset_index(drop=True)
    return df_all


# %%
# ---------------------------
# UPDATED Feature Engineering (Dynamic indicators)
# ---------------------------
def add_features(df_all):
    df_all = df_all.copy()

    def fe_group(g):
        g = g.sort_values("Date").reset_index(drop=True)
        rows = len(g)

        print(f"[{g['symbol'].iloc[0]}] Rows available = {rows}")

        # ================================
        # PRIORITY 1 → FULL INDICATORS
        # Need 50 rows (MA50 requires 50)
        # ================================
        if rows >= 50:
            print(" → Using FULL indicators (RSI14, MA20, MA50)...")

            g["RSI14"] = compute_rsi(g["Close"], 14)
            g["MA20"] = g["Close"].rolling(20).mean()
            g["MA50"] = g["Close"].rolling(50).mean()

            # lag features
            g["Close_1"] = g["Close"].shift(1)
            g["Close_2"] = g["Close"].shift(2)
            g["Close_3"] = g["Close"].shift(3)
            g["Close_5"] = g["Close"].shift(5)

        # ================================
        # PRIORITY 2 → MEDIUM INDICATORS
        # Need ≥ 20 rows
        # ================================
        elif rows >= 20:
            print(" → Using MEDIUM indicators (RSI7, MA10)...")

            g["RSI7"] = compute_rsi(g["Close"], 7)
            g["MA10"] = g["Close"].rolling(10).mean()

            g["Close_1"] = g["Close"].shift(1)
            g["Close_2"] = g["Close"].shift(2)
            g["Close_3"] = g["Close"].shift(3)

        # ================================
        # PRIORITY 3 → SMALL INDICATORS
        # Need ≥ 14 rows
        # ================================
        elif rows >= 14:
            print(" → Using SMALL indicators (RSI7 only + lags)...")

            g["RSI7"] = compute_rsi(g["Close"], 7)

            g["Close_1"] = g["Close"].shift(1)
            g["Close_2"] = g["Close"].shift(2)

        # ================================
        # PRIORITY 4 → MINIMAL FEATURES
        # Need ≥ 7 rows
        # Only lag features
        # ================================
        elif rows >= 7:
            print(" → Using MINIMAL features (lags only)...")

            g["Close_1"] = g["Close"].shift(1)
            g["Close_2"] = g["Close"].shift(2)

        # ================================
        # PRIORITY 5 → TOO LITTLE DATA
        # Skip this stock entirely
        # ================================
        else:
            print(" - Not enough data (< 7 rows), skipping this symbol")
            return pd.DataFrame()

        # Target: next candle direction
        g["target"] = (g["Close"].shift(-1) > g["Close"]).astype(int)

        # Drop unusable rows (indicator warmups)
        g = g.dropna().reset_index(drop=True)
        return g

    # Apply per symbol
    df_fe = df_all.groupby("symbol", group_keys=False).apply(fe_group)

    print("Final feature-engineered shape:", df_fe.shape)
    return df_fe


# %%
# ---------------------------
# Training pipeline (GridFS version)
# ---------------------------
import io


def train_and_save_model(df_all, fs, feature_cols=None):
    if df_all.empty:
        print("No data available to train.")
        return None, None

    if feature_cols is None:
        feature_cols = [
            'Open','High','Low','Close','Volume',
            'return_1','return_3','return_7',
            'sma_5','sma_10','sma_20','ema_10','ema_20',
            'vol_5','vol_10','mom_3','mom_7',
            'vol_change','vol_ratio_5','rsi_14',
            'month','dayofweek','is_quarter_end',
            'close_lag_1','close_lag_2','close_lag_3','close_lag_5',
            'vol_lag_1','vol_lag_2','vol_lag_3','vol_lag_5'
        ]

    # ensure all features exist
    missing = [c for c in feature_cols if c not in df_all.columns]
    if missing:
        print("Warning - missing features, dropping:", missing)
        feature_cols = [c for c in feature_cols if c in df_all.columns]

    X = df_all[feature_cols].copy()
    y = df_all["target"].copy()

    # sort by time
    df_all = df_all.sort_values("Date").reset_index(drop=True)
    split_index = int(len(df_all) * 0.8)

    X_train = X.iloc[:split_index]
    X_valid = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_valid = y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Models & params
    models_to_try = {}
    param_grids = {}

    models_to_try['rf'] = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grids['rf'] = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6, 8],
        'class_weight': [None, 'balanced']
    }

    try:
        from xgboost import XGBClassifier
        models_to_try['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        param_grids['xgb'] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
    except:
        print("xgboost not available - skipping")

    try:
        from lightgbm import LGBMClassifier
        models_to_try['lgbm'] = LGBMClassifier(random_state=42, n_jobs=-1)
        param_grids['lgbm'] = {
            'n_estimators': [100, 200],
            'max_depth': [-1, 6],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    except:
        print("lightgbm not available - skipping")

    print("Models available:", list(models_to_try.keys()))

    tscv = TimeSeriesSplit(n_splits=3)
    best_estimators = {}
    validation_aucs = {}

    # Tune models
    for name, model in models_to_try.items():
        print(f"\nTuning {name} ...")
        grid = RandomizedSearchCV(
            model,
            param_grids.get(name, {}),
            n_iter=8,
            scoring='roc_auc',
            cv=tscv,
            n_jobs=-1,
            random_state=42
        )
        grid.fit(X_train_scaled, y_train)

        best = grid.best_estimator_
        best_estimators[name] = best

        # predict proba
        if hasattr(best, "predict_proba"):
            y_valid_proba = best.predict_proba(X_valid_scaled)[:, 1]
        else:
            try:
                y_valid_proba = best.decision_function(X_valid_scaled)
                y_valid_proba = (y_valid_proba - y_valid_proba.min()) / (y_valid_proba.max() - y_valid_proba.min() + 1e-8)
            except:
                y_valid_proba = best.predict(X_valid_scaled)

        auc = roc_auc_score(y_valid, y_valid_proba)
        validation_aucs[name] = auc

        print(f" Best params ({name}):", grid.best_params_)
        print(f" Validation AUC ({name}): {auc:.4f}")

    # Choose top models for stacked ensemble
    sorted_models = sorted(validation_aucs.items(), key=lambda x: x[1], reverse=True)
    top_names = [name for name, _ in sorted_models[:3]]

    print("\nStacking:", top_names)

    estimators_for_stack = [(name, best_estimators[name]) for name in top_names]

    stack = StackingClassifier(
        estimators=estimators_for_stack,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    print("\nTraining stacking model...")
    stack.fit(X_train_scaled, y_train)

    # Validate stacking model
    y_valid_proba_stack = stack.predict_proba(X_valid_scaled)[:, 1]
    auc_stack = roc_auc_score(y_valid, y_valid_proba_stack)
    print(f"\nStacking Validation AUC: {auc_stack:.4f}")

    # ----------------------------------------
    # SAVE MODEL + SCALER TO MONGODB GRIDFS
    # ----------------------------------------

    # Remove old files
    for file in fs.find({"filename": "best_stock_model.pkl"}): fs.delete(file._id)
    for file in fs.find({"filename": "scaler.pkl"}): fs.delete(file._id)

    # Save new files
    model_bytes = io.BytesIO()
    scaler_bytes = io.BytesIO()

    joblib.dump(stack, model_bytes)
    joblib.dump(scaler, scaler_bytes)

    model_bytes.seek(0)
    scaler_bytes.seek(0)

    fs.put(model_bytes.read(), filename="best_stock_model.pkl")
    fs.put(scaler_bytes.read(), filename="scaler.pkl")

    print("\n Model + scaler saved to MongoDB GridFS")

    return stack, scaler


# %%
# ---------------------------
# Orchestration: load -> fe -> train (GridFS version)
# ---------------------------
def retrain_from_mongo(days=LOOKBACK_DAYS):
    # Connect to MongoDB
    client = connect_mongo(MONGODB_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)

    # Load recent data
    raw = load_last_n_days_from_mongo(client, DB_NAME, STOCK_LIST, days=days)
    if raw.empty:
        print("No data loaded from MongoDB. Aborting retrain.")
        return None, None

    # Feature engineering
    df_fe = add_features(raw)
    print("Feature-engineered dataset shape:", df_fe.shape)

    # Train & save model -> GridFS
    stack, scaler = train_and_save_model(df_fe, fs)

    print("Retraining complete.")
    return stack, scaler


# %%
# ---------------------------
# Optional: Background retrain scheduler (every N days)
# ---------------------------
def start_auto_retrain(every_n_days=30, run_immediately=False):
    """
    Starts a background scheduler that runs retrain_from_mongo every `every_n_days`.
    Works with GridFS model saving. Not recommended on short-lived hosting
    (Render free plan resets dynos).
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=lambda: retrain_from_mongo(days=LOOKBACK_DAYS),
        trigger='interval',
        days=every_n_days,
        next_run_time=(datetime.now() if run_immediately else None)
    )

    scheduler.start()

    print(f"[Scheduler] Auto-retrain scheduled every {every_n_days} days.")
    if run_immediately:
        print("[Scheduler] First retrain will run immediately.")

    return scheduler


# %%
# ---------------------------
# Run a single retrain now (manual trigger)
# ---------------------------
if __name__ == "__main__":
    print("Starting single retrain job...")
    model, scaler = retrain_from_mongo(days=LOOKBACK_DAYS)
    print("Retrain complete.")



