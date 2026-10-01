import os
import time
import hmac
import hashlib
import json
import requests
import joblib

from datetime import datetime
from flask import Flask, jsonify

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# =========================================================
# CONFIGURATION
# =========================================================
BASE_URL = os.environ.get("DELTA_BASE_URL", "https://api.india.delta.exchange")
API_KEY = os.environ.get("DELTA_API_KEY", "")
API_SECRET = os.environ.get("DELTA_API_SECRET", "")

SYMBOL = "BTCUSD"
PRODUCT_ID = 27

MODEL_FILE = "ai_brain_model.pkl"
MEMORY_FILE = "trade_memory.json"

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
CONFIDENCE_BASE_THRESHOLD = 0.60
MARGIN_ALLOCATION_PERCENT = 0.05
MAX_LEVERAGE = 5

FEATURES = [
    "return", "ma7", "ma25", "volume", "atr", "rsi",
    "lower_wick_ratio", "upper_wick_ratio", "vol_surge",
    "norm_atr", "wick_skew", "dist_ma25"
]

# =========================================================
# AUTHENTICATION
# =========================================================
def generate_signature(secret, method, path, query="", payload=""):
    timestamp = str(int(time.time()))
    message = method.upper() + timestamp + path + query + payload
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return signature, timestamp

def get_headers(method, path, query="", payload=""):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Delta API credentials are missing.")
    signature, timestamp = generate_signature(API_SECRET, method, path, query, payload)
    return {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "BTC-AI-Trading-Bot/1.0"
    }

# =========================================================
# 1. MEMORY ENGINE
# =========================================================
def default_memory():
    return {"loss_penalty": 0.0, "total_trades": 0, "losses": 0, "wins": 0}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return default_memory()
    try:
        with open(MEMORY_FILE, "r") as f:
            base = default_memory()
            base.update(json.load(f))
            return base
    except Exception as e:
        print(f"[MEMORY ERROR] {e}")
        return default_memory()

def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")

# =========================================================
# 2. PRODUCT SPECS & TICK ROUNDING
# =========================================================
def get_product_specs():
    endpoints = [
        f"{BASE_URL}/v2/products/{SYMBOL}",
        f"https://api.delta.exchange/v2/products/{SYMBOL}"
    ]
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5, headers={"Accept": "application/json"})
            data = res.json()
            if data.get("success") and data.get("result"):
                product = data.get("result")
                return {
                    "contract_value": float(product.get("contract_value", 0.001)),
                    "tick_size": float(product.get("tick_size", 0.5))
                }
        except Exception:
            continue
    return {"contract_value": 0.001, "tick_size": 0.5}

def round_to_tick(price, tick_size):
    if tick_size <= 0:
        return float(price)
    return round(round(price / tick_size) * tick_size, 8)

# =========================================================
# 3. WALLET BALANCE & MARKET DATA
# =========================================================
def get_available_balance():
    try:
        path = "/v2/wallet/balances"
        headers = get_headers("GET", path)
        res = requests.get(BASE_URL + path, headers=headers, timeout=10)
        data = res.json()
        if not data.get("success"):
            return 0.0
            
        for item in data.get("result", []):
            if item.get("asset_symbol") in ["USD", "USDT", "INR"]:
                bal = float(item.get("available_balance", 0.0))
                if bal > 0:
                    return bal
        return 0.0
    except Exception as e:
        print(f"[BALANCE ERROR] {e}")
        return 0.0

def fetch_market_data():
    try:
        now = int(time.time())
        lookback_seconds = 120 * 15 * 60
        start_time = now - lookback_seconds

        base_urls = []
        if BASE_URL:
            base_urls.append(BASE_URL.rstrip("/"))

        india_url = "https://api.india.delta.exchange"
        if india_url not in base_urls:
            base_urls.append(india_url)

        global_url = "https://api.delta.exchange"
        if global_url not in base_urls:
            base_urls.append(global_url)

        base_urls = list(dict.fromkeys(base_urls))

        candles = None
        successful_url = None

        for base in base_urls:
            url = f"{base}/v2/history/candles"
            params = {
                "resolution": "15m",
                "symbol": SYMBOL,
                "start": start_time,
                "end": now
            }

            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=15,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "TradingBot/1.0"
                    }
                )

                if response.status_code != 200:
                    continue

                data = response.json()
                if data.get("success") is False:
                    continue

                candles = data.get("result")
                if candles and isinstance(candles, list) and len(candles) > 0:
                    successful_url = url
                    break
            except Exception as e:
                print(f"[API ATTEMPT ERROR] {e}")
                continue

        if not candles:
            print("[MARKET ERROR] Candles data bilkul mali nathi rahyo.")
            return None

        df = pd.DataFrame(candles)
        rename_map = {
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
        }
        df = df.rename(columns=rename_map)

        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_columns).copy()

        if "time" in df.columns:
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])
            df = df.sort_values("time")
            df = df.drop_duplicates(subset=["time"], keep="last")

        df = df.reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        return None

def fetch_order_book_metrics():
    endpoints = [
        f"{BASE_URL}/v2/l2orderbook/{SYMBOL}",
        f"https://api.india.delta.exchange/v2/l2orderbook/{SYMBOL}",
        f"https://api.delta.exchange/v2/l2orderbook/{SYMBOL}"
    ]
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5)
            data = res.json()
            if data.get("success"):
                book = data.get("result", {})
                bids = book.get("buy", [])
                asks = book.get("sell", [])
                if bids and asks:
                    top_bids = sum(float(x.get("size", 0)) for x in bids[:10])
                    top_asks = sum(float(x.get("size", 0)) for x in asks[:10])
                    total = top_bids + top_asks + 1e-9
                    obi = (top_bids - top_asks) / total
                    return obi, top_bids, top_asks
        except Exception:
            continue
    return 0.0, 0.0, 0.0

# =========================================================
# 4. QUANT INDICATORS
# =========================================================
def add_indicators(df):
    df = df.copy()
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma25"] = df["close"].rolling(25).mean()
    df["return"] = df["close"].pct_change()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    candle_range = (df["high"] - df["low"]).replace(0, 1e-9)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    df["lower_wick_ratio"] = (body_low - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - body_high) / candle_range

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_surge"] = df["volume"] / (df["vol_ma20"] + 1e-9)

    df["norm_atr"] = df["atr"] / (df["close"] + 1e-9)
    df["wick_skew"] = df["lower_wick_ratio"] - df["upper_wick_ratio"]
    df["dist_ma25"] = (df["close"] - df["ma25"]) / (df["ma25"] + 1e-9)

    df["recent_low"] = df["low"].shift(1).rolling(20).min()
    df["recent_high"] = df["high"].shift(1).rolling(20).max()
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    return df

# =========================================================
# 5. WHALE-TRAP SCANNER
# =========================================================
def check_institutional_sweep(df):
    if len(df) < 30:
        return None
    latest = df.iloc[-1]
    atr = float(latest["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    obi_ratio, _, _ = fetch_order_book_metrics()
    tick_size = get_product_specs()["tick_size"]

    if latest["low"] < latest["recent_low"] and latest["close"] > latest["recent_low"]:
        if obi_ratio > 0.35 and latest["lower_wick_ratio"] >= 0.40 and latest["vol_surge"] >= 1.20:
            sl = round_to_tick(latest["low"] - 0.4 * atr, tick_size)
            tp = round_to_tick(latest["close"] + 3.0 * atr, tick_size)
            return ("BUY", 0.92, sl, tp, "INSTITUTIONAL_BULLISH_SWEEP")

    if latest["high"] > latest["recent_high"] and latest["close"] < latest["recent_high"]:
        if obi_ratio < -0.35 and latest["upper_wick_ratio"] >= 0.40 and latest["vol_surge"] >= 1.20:
            sl = round_to_tick(latest["high"] + 0.4 * atr, tick_size)
            tp = round_to_tick(latest["close"] - 3.0 * atr, tick_size)
            return ("SELL", 0.92, sl, tp, "INSTITUTIONAL_BEARISH_SWEEP")

    return None

# =========================================================
# 6. AI BRAIN PIPELINE & AUTO RE-TRAINING
# =========================================================
def build_ai_pipeline():
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=42
        ))
    ])

def train_and_save_ai_brain():
    print(f"\n[{datetime.now()}] [AI] Auto Re-training sharu thay che...")
    df = fetch_market_data()
    if df is None or len(df) < 30:
        return None

    df = add_indicators(df)
    df_clean = df.dropna(subset=FEATURES + ["target"]).copy()
    if len(df_clean) < 25:
        return None

    X = df_clean[FEATURES][:-1]
    y = df_clean["target"][:-1]

    pipeline = build_ai_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"[{datetime.now()}] [AI] Model save thai gayu: {MODEL_FILE}")
    return pipeline

def get_or_load_ai_brain():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            print(f"[AI LOAD ERROR] {e}")
    return train_and_save_ai_brain()

# =========================================================
# 7. POSITION SIZING
# =========================================================
def calculate_contracts(balance, leverage, entry_price):
    specs = get_product_specs()
    contract_val = specs["contract_value"]
    if balance <= 0 or entry_price <= 0:
        return 0
    margin = balance * MARGIN_ALLOCATION_PERCENT
    notional = margin * leverage
    contract_notional = entry_price * contract_val
    if contract_notional <= 0:
        return 0
    return max(1, int(notional / contract_notional))

# =========================================================
# 8. PREDICTION ENGINE
# =========================================================
def predict_signal(df):
    if df is None or len(df) < 30:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_DATA")

    df = add_indicators(df)
    df_clean = df.dropna(subset=FEATURES).copy()
    if len(df_clean) < 20:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_FEATURE_DATA")

    inst_signal = check_institutional_sweep(df_clean)
    if inst_signal:
        action, conf, sl, tp, tag = inst_signal
        return (action, conf, sl, tp, 4, tag)

    pipeline = get_or_load_ai_brain()
    if pipeline is None:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "AI_TRAIN_FAIL")

    latest_features = df_clean[FEATURES].iloc[[-1]]
    probabilities = pipeline.predict_proba(latest_features)[0]
    classes = pipeline.named_steps["model"].classes_

    prob_up, prob_down = 0.0, 0.0
    for cls, prob in zip(classes, probabilities):
        if int(cls) == 1:
            prob_up = float(prob)
        else:
            prob_down = float(prob)

    memory = load_memory()
    threshold = min(0.90, CONFIDENCE_BASE_THRESHOLD + memory.get("loss_penalty", 0.0))
    current_atr = float(df_clean["atr"].iloc[-1])
    latest_close = float(df_clean["close"].iloc[-1])

    if prob_up >= threshold:
        action = "BUY"
        conf = prob_up
    elif prob_down >= threshold:
        action = "SELL"
        conf = prob_down
    else:
        return ("HOLD", max(prob_up, prob_down), 0.0, 0.0, 0, "AI_WAIT_AND_SEE")

    leverage = 5 if conf >= 0.78 else (4 if conf >= 0.68 else (3 if conf >= 0.60 else 2))
    leverage = min(leverage, MAX_LEVERAGE)

    tick_size = get_product_specs()["tick_size"]
    sl_dist = current_atr * 1.2
    tp_dist = current_atr * 2.8

    if action == "BUY":
        sl = round_to_tick(latest_close - sl_dist, tick_size)
        tp = round_to_tick(latest_close + tp_dist, tick_size)
    else:
        sl = round_to_tick(latest_close + sl_dist, tick_size)
        tp = round_to_tick(latest_close - tp_dist, tick_size)

    return (action, conf, sl, tp, leverage, f"AI_BRAIN (UP:{prob_up:.2f}, DOWN:{prob_down:.2f})")

# =========================================================
# 9. ORDER EXECUTION
# =========================================================
def place_order_with_brackets(action, size, stop_loss, take_profit):
    if DRY_RUN:
        return {"status": "DRY_RUN_SUCCESS", "message": "Dry run active. No real order placed."}
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

# =========================================================
# 10. FLASK ROUTES & SCHEDULER
# =========================================================
@app.route("/", methods=["GET"])
@app.route("/execute-trade", methods=["GET"])
def execute_trade():
    df = fetch_market_data()
    action, conf, sl, tp, leverage, strategy_tag = predict_signal(df)
    balance = get_available_balance()
    latest_close = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0

    if action in ["BUY", "SELL"]:
        contracts = calculate_contracts(balance, leverage, latest_close)
        if balance <= 0 and not DRY_RUN:
            return jsonify({
                "status": "FAILED_NO_BALANCE",
                "strategy": strategy_tag,
                "action": action,
                "confidence": f"{conf*100:.2f}%",
                "balance": f"₹{balance:.2f}"
            }), 200

        order_res = place_order_with_brackets(action, contracts, sl, tp)
        return jsonify({
            "status": "ORDER_PLACED" if not DRY_RUN else "SIMULATED_ORDER",
            "action": action,
            "confidence": f"{conf*100:.2f}%",
            "contracts": contracts,
            "leverage": f"{leverage}x",
            "stop_loss": sl,
            "take_profit": tp,
            "strategy": strategy_tag,
            "delta_response": order_res
        }), 200

    return jsonify({
        "status": "WAIT_AND_SEE",
        "action": "HOLD",
        "confidence": f"{conf*100:.2f}%",
        "strategy": strategy_tag,
        "balance": f"₹{balance:.2f}"
    }), 200

scheduler = BackgroundScheduler()
scheduler.add_job(func=train_and_save_ai_brain, trigger="cron", day_of_week="sun", hour=0, minute=0)
scheduler.start()

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        train_and_save_ai_brain()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    
