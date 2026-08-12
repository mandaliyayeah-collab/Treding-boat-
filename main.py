import os
import time
import hmac
import hashlib
import json
import requests
import threading
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ============================================================
# DELTA EXCHANGE TESTNET CONFIGURATION
# ============================================================
BASE_URL = "https://cdn-ind.testnet.deltaex.org"
TRADING_URL = "https://testnet-api.delta.exchange"

SYMBOL = "BTCUSD"
RESOLUTION = "1h"

# API Keys from Screenshot
API_KEY = "3XVMBQOHeYoxaMEa863eJDxdt15XKp"
API_SECRET = "5Heph59t27suDj0jqUd18Mb8uoAv5OCB2yDC5FrhvLO7bNPpmyiANHIdJ8Cy"

# ============================================================
# SIGNATURE GENERATOR
# ============================================================
def generate_signature(method, endpoint, payload_str, timestamp):
    message = method + timestamp + endpoint + payload_str
    return hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

# ============================================================
# GET HISTORICAL CANDLES
# ============================================================
def get_candles():
    end_time = int(time.time())
    start_time = end_time - (500 * 60 * 60)
    url = f"{BASE_URL}/v2/history/candles"

    params = {
        "resolution": RESOLUTION,
        "symbol": SYMBOL,
        "start": start_time,
        "end": end_time
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise Exception(f"Delta API error: {data}")

    candles = data.get("result", [])
    if not candles:
        raise Exception("No candle data received from Delta Exchange")

    df = pd.DataFrame(candles)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ============================================================
# TECHNICAL INDICATORS & DATA PREPARATION
# ============================================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window).mean()
    average_loss = loss.rolling(window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    df = df.copy()
    df["rsi"] = calculate_rsi(df["close"])
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(10).std()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)

# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================
def make_prediction():
    df = get_candles()
    df = prepare_data(df)

    features = ["rsi", "sma_10", "sma_20", "ema_10", "ema_20", "return", "volatility"]

    if len(df) < 100:
        raise Exception(f"Not enough data for ML. Only {len(df)} rows available.")

    train_df = df.iloc[:-1].copy()
    X = train_df[features]
    y = train_df["target"]

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8, min_samples_leaf=3)
    model.fit(X, y)

    latest = df.iloc[-1]
    X_latest = df[features].iloc[[-1]]

    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]

    signal = "BUY" if prediction == 1 else "SELL"

    return {
        "symbol": SYMBOL,
        "timeframe": RESOLUTION,
        "signal": signal,
        "probability_up": round(float(probability[1]) * 100, 2),
        "probability_down": round(float(probability[0]) * 100, 2),
        "price": float(latest["close"]),
        "rsi": round(float(latest["rsi"]), 2),
        "candles_used": len(df)
    }

# ============================================================
# PLACE ORDER ON DELTA TESTNET
# ============================================================
def place_order(side, price):
    endpoint = "/v2/orders"
    timestamp = str(int(time.time()))
    payload = {
        "product_id": 1,
        "size": 1,
        "side": side.lower(),
        "order_type": "limit_order",
        "limit_price": str(price)
    }
    payload_str = json.dumps(payload)
    signature = generate_signature("POST", endpoint, payload_str, timestamp)

    headers = {
        'api-key': API_KEY,
        'timestamp': timestamp,
        'signature': signature,
        'Content-Type': 'application/json'
    }

    res = requests.post(TRADING_URL + endpoint, data=payload_str, headers=headers)
    return res.json()

# ============================================================
# AUTOMATED BACKGROUND TRADING LOOP (Runs every 30 Mins)
# ============================================================
def auto_trading_bot():
    print("Auto-Trading Bot Loop Started...")
    while True:
        try:
            pred = make_prediction()
            print(f"Checking Market: Signal={pred['signal']}, Price={pred['price']}, RSI={pred['rsi']}")
            # Execute Order if confidence is strong
            if pred['probability_up'] > 60 or pred['probability_down'] > 60:
                res = place_order(pred['signal'], pred['price'])
                print("Auto-Trade Placed Response:", res)
        except Exception as e:
            print(f"Auto-Trade Loop Error: {e}")
        time.sleep(1800)

threading.Thread(target=auto_trading_bot, daemon=True).start()

# ============================================================
# FLASK WEB ENDPOINTS
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Delta Exchange ML API & Auto-Bot active",
        "symbol": SYMBOL
    })

@app.route("/predict")
def predict():
    try:
        result = make_prediction()
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/execute-trade")
def execute_trade_endpoint():
    try:
        pred = make_prediction()
        res = place_order(pred['signal'], pred['price'])
        return jsonify({"success": True, "prediction": pred, "order_response": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
