import os
import time
import hmac
import hashlib
import json
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- DELTA PRODUCTION LIVE CONFIGURATION ---
BASE_URL = "https://api.delta.exchange"
API_KEY = os.environ.get("DELTA_API_KEY", "BJ5AVamwEw6jQBbsTngkdzf5SSPrti")
API_SECRET = os.environ.get("DELTA_API_SECRET", "UhGk2EPyyRxtPW1JLAMMdtjCU4wGnbdyVTE3KJdM5Qk48MeEUqu54apK0LBx")

SYMBOL = "BTCUSD"
CONFIDENCE_THRESHOLD = 0.55  # 55% Confidence Requirement
RISK_PER_TRADE_PERCENT = 0.05  # 5% of available balance per trade (Auto-Compounding)
LEVERAGE = 3  # Safe 3x Leverage

def generate_signature(secret, method, path, query="", payload=""):
    timestamp = str(int(time.time()))
    message = method + timestamp + path + query + payload
    signature = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature, timestamp

def get_headers(method, path, query="", payload=""):
    signature, timestamp = generate_signature(API_SECRET, method, path, query, payload)
    return {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json"
    }

# 1. Fetch live balance for automatic compounding
def get_available_balance():
    try:
        path = "/v2/wallet/balances"
        headers = get_headers("GET", path)
        res = requests.get(BASE_URL + path, headers=headers, timeout=10)
        data = res.json()
        if data.get("success"):
            for item in data.get("result", []):
                if item.get("asset_symbol") in ["USDT", "USD"]:
                    return float(item.get("available_balance", 0))
        return 50.0
    except Exception as e:
        print(f"Balance fetch error: {e}")
        return 50.0

# 2. Fetch live candles
def fetch_market_data():
    try:
        path = f"/v2/chart/history?symbol={SYMBOL}&resolution=5"
        res = requests.get(BASE_URL + path, timeout=10)
        candles = res.json().get("result", [])
        if not candles:
            return None
        
        df = pd.DataFrame(candles)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        print(f"Data fetch error: {e}")
        return None

# 3. AI prediction model
def predict_signal(df):
    if df is None or len(df) < 50:
        return "HOLD", 0.0

    df['return'] = df['close'].pct_change()
    df['ma7'] = df['close'].rolling(7).mean()
    df['ma25'] = df['close'].rolling(25).mean()
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    df_clean = df.dropna()
    features = ['return', 'ma7', 'ma25', 'volume']
    
    X = df_clean[features][:-1]
    y = df_clean['target'][:-1]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    latest_features = df_clean[features].iloc[[-1]]
    probabilities = model.predict_proba(latest_features)[0]
    
    prob_down, prob_up = probabilities[0], probabilities[1]
    
    if prob_up >= CONFIDENCE_THRESHOLD:
        return "BUY", prob_up
    elif prob_down >= CONFIDENCE_THRESHOLD:
        return "SELL", prob_down
    else:
        return "HOLD", max(prob_up, prob_down)

# 4. Place live order on Delta Exchange
def place_order(action, size):
    try:
        path = "/v2/orders"
        payload = json.dumps({
            "product_id": 27,  # BTCUSD Perpetual contract ID
            "size": int(size),
            "side": "buy" if action == "BUY" else "sell",
            "order_type": "market_order"
        })
        headers = get_headers("POST", path, payload=payload)
        res = requests.post(BASE_URL + path, headers=headers, data=payload, timeout=10)
        return res.json()
    except Exception as e:
        print(f"Order error: {e}")
        return {"error": str(e)}

# 5. Endpoint triggered by cron-job.org
@app.route("/execute-trade", methods=["GET"])
def execute_trade():
    df = fetch_market_data()
    action, confidence = predict_signal(df)
    
    if action in ["BUY", "SELL"]:
        balance = get_available_balance()
        trade_margin = balance * RISK_PER_TRADE_PERCENT
        contracts = max(1, int(trade_margin * LEVERAGE))
        
        order_res = place_order(action, contracts)
        return jsonify({
            "status": "ORDER_PLACED",
            "action": action,
            "confidence": f"{confidence * 100:.2f}%",
            "contracts": contracts,
            "order_response": order_res
        }), 200
    
    return jsonify({
        "status": "WAIT_AND_SEE",
        "reason": "Market conditions not matching 55%+ threshold",
        "action": action,
        "confidence": f"{confidence * 100:.2f}%"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    
