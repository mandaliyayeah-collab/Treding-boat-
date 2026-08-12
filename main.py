"        from flask import Flask, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

app = Flask(__name__)

# ============================================================
# DELTA EXCHANGE TESTNET
# ============================================================

BASE_URL = "https://cdn-ind.testnet.deltaex.org"

SYMBOL = "BTCUSD"
RESOLUTION = "1h"


# ============================================================
# GET HISTORICAL CANDLES
# ============================================================

def get_candles():

    end_time = int(time.time())

    # 500 hours approximately
    start_time = end_time - (500 * 60 * 60)

    url = f"{BASE_URL}/v2/history/candles"

    params = {
        "resolution": RESOLUTION,
        "symbol": SYMBOL,
        "start": start_time,
        "end": end_time
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"Delta API error: {data}")

    candles = data.get("result", [])

    if not candles:
        raise Exception("No candle data received from Delta Exchange")

    df = pd.DataFrame(candles)

    # Make sure required columns exist
    required_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise Exception(
                f"Missing column '{column}'. Received: {df.columns.tolist()}"
            )

    # Convert numbers
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    # Sort oldest -> newest
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, window=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window).mean()
    average_loss = loss.rolling(window).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# CREATE FEATURES
# ============================================================

def prepare_data(df):

    df = df.copy()

    df["rsi"] = calculate_rsi(df["close"])

    df["sma_10"] = df["close"].rolling(10).mean()

    df["sma_20"] = df["close"].rolling(20).mean()

    df["ema_10"] = df["close"].ewm(
        span=10,
        adjust=False
    ).mean()

    df["ema_20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["return"] = df["close"].pct_change()

    df["volatility"] = (
        df["return"]
        .rolling(10)
        .std()
    )

    # Next candle direction
    df["future_close"] = df["close"].shift(-1)

    df["target"] = (
        df["future_close"] > df["close"]
    ).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

def make_prediction():

    df = get_candles()

    df = prepare_data(df)

    features = [
        "rsi",
        "sma_10",
        "sma_20",
        "ema_10",
        "ema_20",
        "return",
        "volatility"
    ]

    if len(df) < 100:
        raise Exception(
            f"Not enough data for ML. Only {len(df)} rows available."
        )

    # Keep last row separate for prediction
    train_df = df.iloc[:-1].copy()

    X = train_df[features]
    y = train_df["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_leaf=3
    )

    model.fit(X, y)

    latest = df.iloc[-1]

    X_latest = df[features].iloc[[-1]]

    prediction = model.predict(X_latest)[0]

    probability = model.predict_proba(
        X_latest
    )[0]

    probability_up = float(probability[1])

    probability_down = float(probability[0])

    if prediction == 1:
        signal = "BUY"
    else:
        signal = "SELL"

    return {
        "symbol": SYMBOL,
        "timeframe": RESOLUTION,
        "signal": signal,
        "probability_up": round(
            probability_up * 100,
            2
        ),
        "probability_down": round(
            probability_down * 100,
            2
        ),
        "price": float(latest["close"]),
        "rsi": round(
            float(latest["rsi"]),
            2
        ),
        "sma_10": round(
            float(latest["sma_10"]),
            2
        ),
        "sma_20": round(
            float(latest["sma_20"]),
            2
        ),
        "ema_10": round(
            float(latest["ema_10"]),
            2
        ),
        "ema_20": round(
            float(latest["ema_20"]),
            2
        ),
        "candles_used": len(df)
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "running",
        "message": "Delta Exchange ML API is working",
        "symbol": SYMBOL,
        "timeframe": RESOLUTION
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict")
def predict():

    try:

        result = make_prediction()

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
