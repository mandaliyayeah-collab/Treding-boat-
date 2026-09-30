import os
import time
import hmac
import hashlib
import json
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# --- DELTA LIVE CONFIGURATION ---
BASE_URL = "https://api.delta.exchange"
API_KEY = os.environ.get("DELTA_API_KEY", "BJ5AVamwEw6jQBbsTngkdzf5SSPrti")
API_SECRET = os.environ.get("DELTA_API_SECRET", "UhGk2EPyyRxtPW1JLAMMdtjCU4wGnbdyVTE3KJdM5Qk48MeEUqu54apK0LBx")

SYMBOL = "BTCUSD"
PRODUCT_ID = 27
CONFIDENCE_BASE_THRESHOLD = 0.55
RISK_PER_TRADE_PERCENT = 0.05
MEMORY_FILE = "trade_memory.json"

# --- AUTHENTICATION ---
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

# --- 1. SELF-LEARNING MEMORY ENGINE ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"loss_penalty": 0.0, "total_trades": 0, "losses": 0, "wins": 0}
    return {"loss_penalty": 0.0, "total_trades": 0, "losses": 0, "wins": 0}

def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[MEMORY ERROR] {e}")

# --- 2. LIVE DATA & TECHNICAL INDICATORS ---
def get_available_balance():
    try:
        path = "/v2/wallet/balances"
        headers = get_headers("GET", path)
        res = requests.get(BASE_URL + path, headers=headers, timeout=10)
        data = res.json()
        if data.get("success"):
            for item in data.get("result", []):
                if item.get("asset_symbol") in ["USDT", "USD"]:
                    return float(item.get("available_balance", 0.0))
        return 0.0
    except Exception as e:
        print(f"[ERROR] Balance fetch error: {e}")
        return 0.0

def fetch_market_data():
    try:
        path = f"/v2/chart/history?symbol={SYMBOL}&resolution=15"
        res = requests.get(BASE_URL + path, timeout=10)
        candles = res.json().get("result", [])
        if not candles:
            return None
        
        df = pd.DataFrame(candles)
        for col in ['close', 'open', 'high', 'low', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"[ERROR] Market data error: {e}")
        return None

def add_indicators(df):
    df['ma7'] = df['close'].rolling(7).mean()
    df['ma25'] = df['close'].rolling(25).mean()
    df['return'] = df['close'].pct_change()
    
    # ATR (Volatility)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- WHALE & LIQUIDITY SWEEP FEATURES ---
    candle_range = (df['high'] - df['low']).replace(0, 1e-9)
    df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] / (df['vol_ma20'] + 1e-9)
    
    # 20-period Support & Resistance (excluding current candle)
    df['recent_low'] = df['low'].shift(1).rolling(20).min()
    df['recent_high'] = df['high'].shift(1).rolling(20).max()
    
    # Target definition for AI
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df

# --- 3. WHALE-TRAP SCANNER ---
def check_whale_sweep(df_clean):
    latest = df_clean.iloc[-1]
    atr = float(latest['atr'])
    
    # Bullish Liquidity Sweep (Whale Buy after hunting stop-losses below support)
    if latest['low'] < latest['recent_low'] and latest['close'] > latest['recent_low']:
        if latest['lower_wick_ratio'] >= 0.45 and latest['vol_surge'] >= 1.3:
            sl = round(latest['low'] - (0.5 * atr), 1)
            tp = round(latest['close'] + (2.5 * atr), 1)
            return "BUY", 0.88, sl, tp, 4, "WHALE_BULLISH_SWEEP"

    # Bearish Liquidity Sweep (Whale Sell after hunting stop-losses above resistance)
    if latest['high'] > latest['recent_high'] and latest['close'] < latest['recent_high']:
        if latest['upper_wick_ratio'] >= 0.45 and latest['vol_surge'] >= 1.3:
            sl = round(latest['high'] + (0.5 * atr), 1)
            tp = round(latest['close'] - (2.5 * atr), 1)
            return "SELL", 0.88, sl, tp, 4, "WHALE_BEARISH_SWEEP"

    return None

# --- 4. ADAPTIVE AI PREDICTION ---
def predict_signal(df):
    if df is None or len(df) < 60:
        return "HOLD", 0.0, 0.0, 0.0, 2, "INSUFFICIENT_DATA"
    
    df = add_indicators(df)
    df_clean = df.dropna()
    
    # Priority 1: Check Whale Liquidity Trap
    whale_signal = check_whale_sweep(df_clean)
    if whale_signal:
        return whale_signal
    
    # Priority 2: Gradient Boosting Model with Sweep Features
    features = ['return', 'ma7', 'ma25', 'volume', 'atr', 'rsi', 'lower_wick_ratio', 'upper_wick_ratio', 'vol_surge']
    X = df_clean[features][:-1]
    y = df_clean['target'][:-1]
    
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42)
    model.fit(X, y)
    
    latest_features = df_clean[features].iloc[[-1]]
    probabilities = model.predict_proba(latest_features)[0]
    prob_down, prob_up = float(probabilities[0]), float(probabilities[1])
    
    memory = load_memory()
    dynamic_threshold = CONFIDENCE_BASE_THRESHOLD + memory.get("loss_penalty", 0.0)
    
    current_atr = float(df_clean['atr'].iloc[-1])
    latest_close = float(df_clean['close'].iloc[-1])
    
    if prob_up >= dynamic_threshold:
        action = "BUY"
        conf = prob_up
    elif prob_down >= dynamic_threshold:
        action = "SELL"
        conf = prob_down
    else:
        return "HOLD", max(prob_up, prob_down), 0.0, 0.0, 2, "AI_WAIT_AND_SEE"
    
    if conf >= 0.75:
        leverage = 5
    elif conf >= 0.65:
        leverage = 4
    elif conf >= 0.58:
        leverage = 3
    else:
        leverage = 2
        
    sl_distance = current_atr * 1.5
    tp_distance = current_atr * 3.0
    
    if action == "BUY":
        stop_loss = round(latest_close - sl_distance, 1)
        take_profit = round(latest_close + tp_distance, 1)
    else:
        stop_loss = round(latest_close + sl_distance, 1)
        take_profit = round(latest_close - tp_distance, 1)
        
    return action, conf, stop_loss, take_profit, leverage, "AI_TREND_FOLLOW"

# --- 5. ORDER EXECUTION ---
def place_order_with_brackets(action, size, stop_loss, take_profit):
    try:
        path = "/v2/orders"
        payload = {
            "product_id": PRODUCT_ID,
            "size": int(size),
            "side": "buy" if action == "BUY" else "sell",
            "order_type": "market_order",
            "stop_loss_price": str(stop_loss),
            "take_profit_price": str(take_profit)
        }
        payload_str = json.dumps(payload)
        headers = get_headers("POST", path, payload=payload_str)
        res = requests.post(BASE_URL + path, headers=headers, data=payload_str, timeout=10)
        return res.json()
    except Exception as e:
        print(f"[ORDER ERROR] {e}")
        return {"error": str(e)}

# --- 6. ENDPOINT TRIGGER ---
@app.route("/execute-trade", methods=["GET"])
def execute_trade():
    print("\n================= [AI & WHALE TRACKER START] =================")
    df = fetch_market_data()
    action, confidence, sl, tp, leverage, strategy_tag = predict_signal(df)
    balance = get_available_balance()
    memory = load_memory()
    
    print(f"[STRATEGY] Triggered: {strategy_tag} | Signal: {action}")
    print(f"[CONFIDENCE] {confidence*100:.2f}% | Dynamic Leverage: {leverage}x")
    print(f"[RISK CONTROL] SL: {sl} | Target: {tp} | Penalty: +{memory.get('loss_penalty', 0.0)*100:.2f}%")
    print(f"[WALLET] Live Balance: ${balance:.2f}")
    
    if action in ["BUY", "SELL"]:
        if balance <= 0:
            print("[ABORT] Cancelled: Wallet Balance $0 che.")
            return jsonify({
                "status": "FAILED_NO_BALANCE",
                "strategy": strategy_tag,
                "action": action,
                "confidence": f"{confidence * 100:.2f}%",
                "balance": f"${balance:.2f}"
            }), 200
            
        trade_margin = balance * RISK_PER_TRADE_PERCENT
        contracts = max(1, int(trade_margin * leverage))
        
        print(f"[SUBMITTING] {action} {contracts} Contracts at {leverage}x Leverage...")
        order_res = place_order_with_brackets(action, contracts, sl, tp)
        print(f"[DELTA RESPONSE] {order_res}")
        
        return jsonify({
            "status": "ORDER_PLACED",
            "strategy": strategy_tag,
            "action": action,
            "confidence": f"{confidence * 100:.2f}%",
            "contracts": contracts,
            "leverage": f"{leverage}x",
            "stop_loss": sl,
            "take_profit": tp,
            "order_response": order_res
        }), 200
        
    return jsonify({
        "status": "WAIT_AND_SEE",
        "strategy": strategy_tag,
        "action": "HOLD",
        "confidence": f"{confidence * 100:.2f}%",
        "balance": f"${balance:.2f}"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    
