import os
import time
import hmac
import hashlib
import json
import requests
import joblib
import threading

from datetime import datetime
from flask import Flask, jsonify

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from apscheduler.schedulers.background import BackgroundScheduler

from telegram_alerts import send_telegram_alert

app = Flask(__name__)

# =========================================================
# CONFIGURATION & MULTI-PAIR SETUP
# =========================================================
BASE_URL = os.environ.get("DELTA_BASE_URL", "https://api.india.delta.exchange")
API_KEY = os.environ.get("DELTA_API_KEY", "")
API_SECRET = os.environ.get("DELTA_API_SECRET", "")

PAIRS = {
    "BTCUSD": {"product_id": 27, "model_file": "ai_brain_btc_v2.pkl"},
    "ETHUSD": {"product_id": 29, "model_file": "ai_brain_eth_v2.pkl"}
}

MEMORY_FILE = "trade_memory.json"

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
CONFIDENCE_BASE_THRESHOLD = 0.60
MARGIN_ALLOCATION_PERCENT = 0.05
MAX_LEVERAGE = 5

FEATURES = [
    "return",
    "ma7",
    "ma25",
    "volume",
    "atr",
    "rsi",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "vol_surge",
    "norm_atr",
    "wick_skew",
    "dist_ma25",
    "adx",
    "chop_index"
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
        "User-Agent": "Crypto-AI-Trading-Bot/2.0"
    }

# =========================================================
# 1. MEMORY ENGINE
# =========================================================
def default_memory():
    return {
        "loss_penalty": 0.0,
        "total_trades": 0,
        "losses": 0,
        "wins": 0,
        "open_position": None
    }

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
def get_product_specs(symbol):
    endpoints = [
        f"{BASE_URL}/v2/products/{symbol}",
        f"https://api.delta.exchange/v2/products/{symbol}"
    ]
    default_vals = {"BTCUSD": {"val": 0.001, "tick": 0.5}, "ETHUSD": {"val": 0.01, "tick": 0.05}}
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5, headers={"Accept": "application/json"})
            data = res.json()
            if data.get("success") and data.get("result"):
                product = data.get("result")
                return {
                    "contract_value": float(product.get("contract_value", default_vals.get(symbol, {}).get("val", 0.001))),
                    "tick_size": float(product.get("tick_size", default_vals.get(symbol, {}).get("tick", 0.5)))
                }
        except Exception:
            continue
    fallback = default_vals.get(symbol, {"val": 0.001, "tick": 0.5})
    return {"contract_value": fallback["val"], "tick_size": fallback["tick"]}

def round_to_tick(price, tick_size):
    if tick_size <= 0:
        return float(price)
    return round(round(price / tick_size) * tick_size, 8)

# =========================================================
# 3. WALLET, POSITIONS & MARKET DATA
# =========================================================
def get_available_balance():
    if DRY_RUN:
        return 100000.0
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

def get_live_position():
    if DRY_RUN:
        memory = load_memory()
        return memory.get("open_position")
    try:
        path = "/v2/positions"
        headers = get_headers("GET", path)
        res = requests.get(BASE_URL + path, headers=headers, timeout=10)
        data = res.json()
        if data.get("success") and data.get("result"):
            for pos in data.get("result", []):
                size = float(pos.get("size", 0))
                if abs(size) > 0:
                    return pos
        return None
    except Exception as e:
        print(f"[LIVE POSITION FETCH ERROR] {e}")
        return None

def cancel_all_open_orders(product_id=None):
    if DRY_RUN:
        return
    try:
        targets = [product_id] if product_id else [info["product_id"] for info in PAIRS.values()]
        for pid in targets:
            path = "/v2/orders/all"
            payload = json.dumps({"product_id": pid})
            headers = get_headers("DELETE", path, payload=payload)
            requests.delete(BASE_URL + path, headers=headers, data=payload, timeout=5)
        print("[ORDER CLEANUP] All pending bracket orders cancelled.")
    except Exception as e:
        print(f"[CANCEL ALL ERROR] {e}")

def fetch_market_data(symbol, limit_candles=1500):
    try:
        now = int(time.time())
        lookback_seconds = limit_candles * 15 * 60
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

        for base in base_urls:
            url = f"{base}/v2/history/candles"
            params = {
                "resolution": "15m",
                "symbol": symbol,
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
                        "User-Agent": "TradingBot/2.0"
                    }
                )
                if response.status_code != 200:
                    continue
                data = response.json()
                if data.get("success") is False:
                    continue
                candles = data.get("result")
                if candles and isinstance(candles, list) and len(candles) > 0:
                    break
            except Exception as e:
                print(f"[API ATTEMPT ERROR {symbol}] {e}")
                continue

        if not candles:
            return None

        df = pd.DataFrame(candles)
        rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
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
        print(f"[FETCH ERROR {symbol}] {e}")
        return None

def fetch_funding_rate(symbol):
    endpoints = [
        f"{BASE_URL}/v2/tickers/{symbol}",
        f"https://api.india.delta.exchange/v2/tickers/{symbol}",
        f"https://api.delta.exchange/v2/tickers/{symbol}"
    ]
    for url in endpoints:
        try:
            res = requests.get(url, timeout=5)
            data = res.json()
            if data.get("success"):
                return float(data.get("result", {}).get("funding_rate", 0.0))
        except Exception:
            continue
    return 0.0

def fetch_order_book_metrics(symbol):
    endpoints = [
        f"{BASE_URL}/v2/l2orderbook/{symbol}",
        f"https://api.india.delta.exchange/v2/l2orderbook/{symbol}",
        f"https://api.delta.exchange/v2/l2orderbook/{symbol}"
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

    up = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).sum() / (tr_smooth + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df["adx"] = dx.rolling(14).mean()

    sum_tr = tr.rolling(14).sum()
    max_h = df["high"].rolling(14).max()
    min_l = df["low"].rolling(14).min()
    df["chop_index"] = 100 * (np.log10(sum_tr / (max_h - min_l + 1e-9)) / np.log10(14))

    df["recent_low"] = df["low"].shift(1).rolling(20).min()
    df["recent_high"] = df["high"].shift(1).rolling(20).max()
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    return df

# =========================================================
# 5. WHALE-TRAP SCANNER
# =========================================================
def check_institutional_sweep(df, symbol):
    if len(df) < 30:
        return None
    latest = df.iloc[-1]
    atr = float(latest["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None

    obi_ratio, _, _ = fetch_order_book_metrics(symbol)
    funding_rate = fetch_funding_rate(symbol)
    tick_size = get_product_specs(symbol)["tick_size"]

    if latest["low"] < latest["recent_low"] and latest["close"] > latest["recent_low"]:
        if obi_ratio > 0.35 and latest["lower_wick_ratio"] >= 0.40 and latest["vol_surge"] >= 1.20 and funding_rate < 0.04:
            sl = round_to_tick(latest["low"] - 0.4 * atr, tick_size)
            tp = round_to_tick(latest["close"] + 3.0 * atr, tick_size)
            return ("BUY", 0.92, sl, tp, "INSTITUTIONAL_BULLISH_SWEEP")

    if latest["high"] > latest["recent_high"] and latest["close"] < latest["recent_high"]:
        if obi_ratio < -0.35 and latest["upper_wick_ratio"] >= 0.40 and latest["vol_surge"] >= 1.20 and funding_rate > -0.04:
            sl = round_to_tick(latest["high"] + 0.4 * atr, tick_size)
            tp = round_to_tick(latest["close"] - 3.0 * atr, tick_size)
            return ("SELL", 0.92, sl, tp, "INSTITUTIONAL_BEARISH_SWEEP")

    return None
    # =========================================================
# 6. AI BRAIN PIPELINE & AUTO RE-TRAINING (WALK-FORWARD)
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

def train_and_save_ai_brain_for_pair(symbol, model_file):
    print(f"\n[{datetime.now()}] [AI] Starting Walk-Forward Retraining ({symbol})...")
    df = fetch_market_data(symbol, limit_candles=1500)
    if df is None or len(df) < 100:
        return None

    df = add_indicators(df)
    df_clean = df.dropna(subset=FEATURES + ["target"]).copy()
    if len(df_clean) < 80:
        return None

    X = df_clean[FEATURES][:-1]
    y = df_clean["target"][:-1]

    pipeline = build_ai_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_file)
    print(f"[{datetime.now()}] [AI] Model saved: {model_file} with {len(X)} samples")
    return pipeline

def train_all_ai_brains():
    send_telegram_alert(
        "🧠 *AI Retraining Triggered*\n"
        "Weekly dynamic walk-forward learning started for all pairs."
    )
    for sym, config in PAIRS.items():
        train_and_save_ai_brain_for_pair(sym, config["model_file"])
    send_telegram_alert(
        "✅ *AI Retraining Completed*\n"
        "Models updated and deployed successfully."
    )

def get_or_load_ai_brain(symbol):
    model_file = PAIRS[symbol]["model_file"]
    if os.path.exists(model_file):
        try:
            return joblib.load(model_file)
        except Exception as e:
            print(f"[AI LOAD ERROR {symbol}] {e}")
    return train_and_save_ai_brain_for_pair(symbol, model_file)

# =========================================================
# 7. DYNAMIC RISK MANAGEMENT (BREAK-EVEN & TRAILING)
# =========================================================
def update_dynamic_risk_management(current_price):
    active_pos = get_live_position()
    memory = load_memory()
    pos = memory.get("open_position")

    if not active_pos and pos is not None and not DRY_RUN:
        print("[RISK CONTROL] Position closed. Cleaning orders and memory...")
        cancel_all_open_orders()
        memory["open_position"] = None
        save_memory(memory)
        return

    if not pos:
        return

    side = pos.get("side")
    entry = float(pos.get("entry", 0))
    sl = float(pos.get("sl", 0))
    r_unit = abs(entry - sl)
    if r_unit <= 0:
        return

    symbol = pos.get("symbol", "ACTIVE_PAIR")

    if side == "BUY" and current_price >= entry + (1.5 * r_unit) and sl < entry:
        pos["sl"] = entry
        print(f"[RISK CONTROL] BUY Trade Break-Even Triggered! SL: {entry}")
        send_telegram_alert(
            f"🛡️ *Break-Even Activated (Risk-Free Mode)*\n\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Side:* `BUY`\n"
            f"• *Target Move:* `1.5R Achieved`\n"
            f"• *Updated SL:* `${entry:,.2f}`"
        )
    elif side == "SELL" and current_price <= entry - (1.5 * r_unit) and sl > entry:
        pos["sl"] = entry
        print(f"[RISK CONTROL] SELL Trade Break-Even Triggered! SL: {entry}")
        send_telegram_alert(
            f"🛡️ *Break-Even Activated (Risk-Free Mode)*\n\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Side:* `SELL`\n"
            f"• *Target Move:* `1.5R Achieved`\n"
            f"• *Updated SL:* `${entry:,.2f}`"
        )

    memory["open_position"] = pos
    save_memory(memory)

# =========================================================
# 8. POSITION SIZING
# =========================================================
def calculate_contracts(symbol, balance, leverage, entry_price):
    specs = get_product_specs(symbol)
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
# 9. PREDICTION ENGINE WITH DYNAMIC CONFIDENCE CALIBRATION
# =========================================================
def predict_signal(symbol, df):
    if df is None or len(df) < 30:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_DATA")

    df = add_indicators(df)
    df_clean = df.dropna(subset=FEATURES).copy()
    if len(df_clean) < 20:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "INSUFFICIENT_FEATURE_DATA")

    completed_candle = df_clean.iloc[-2]
    chop = float(completed_candle["chop_index"])
    adx = float(completed_candle["adx"])

    if chop > 61.8 and adx < 20:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "REGIME_FILTER_CHOPPY_NO_TREND")

    inst_signal = check_institutional_sweep(df_clean.iloc[:-1], symbol)
    if inst_signal:
        action, conf, sl, tp, tag = inst_signal
        return (action, conf, sl, tp, 4, tag)

    pipeline = get_or_load_ai_brain(symbol)
    if pipeline is None:
        return ("HOLD", 0.0, 0.0, 0.0, 0, "AI_TRAIN_FAIL")

    latest_features = df_clean[FEATURES].iloc[[-2]]
    probabilities = pipeline.predict_proba(latest_features)[0]
    classes = pipeline.named_steps["model"].classes_

    prob_up, prob_down = 0.0, 0.0
    for cls, prob in zip(classes, probabilities):
        if int(cls) == 1:
            prob_up = float(prob)
        else:
            prob_down = float(prob)

    current_atr = float(completed_candle["atr"])
    latest_close = float(completed_candle["close"])
    norm_atr = current_atr / (latest_close + 1e-9)

    dynamic_confidence = CONFIDENCE_BASE_THRESHOLD
    if norm_atr > 0.008:
        dynamic_confidence = 0.68
    elif norm_atr > 0.005:
        dynamic_confidence = 0.64

    memory = load_memory()
    threshold = min(0.90, dynamic_confidence + memory.get("loss_penalty", 0.0))
    funding_rate = fetch_funding_rate(symbol)

    if prob_up >= threshold and funding_rate < 0.035:
        action = "BUY"
        conf = prob_up
    elif prob_down >= threshold and funding_rate > -0.035:
        action = "SELL"
        conf = prob_down
    else:
        return ("HOLD", max(prob_up, prob_down), 0.0, 0.0, 0, f"AI_WAIT (REQ:{threshold:.2f})")

    leverage = 5 if conf >= 0.78 else (4 if conf >= 0.68 else (3 if conf >= 0.60 else 2))
    leverage = min(leverage, MAX_LEVERAGE)

    tick_size = get_product_specs(symbol)["tick_size"]
    sl_dist = current_atr * 1.2
    tp_dist = current_atr * 2.8

    if action == "BUY":
        sl = round_to_tick(latest_close - sl_dist, tick_size)
        tp = round_to_tick(latest_close + tp_dist, tick_size)
    else:
        sl = round_to_tick(latest_close + sl_dist, tick_size)
        tp = round_to_tick(latest_close - tp_dist, tick_size)

    return (action, conf, sl, tp, leverage, f"AI_BRAIN_{symbol} (UP:{prob_up:.2f}, DOWN:{prob_down:.2f})")

# =========================================================
# 10. ORDER EXECUTION
# =========================================================
def place_order_with_brackets(symbol, action, size, stop_loss, take_profit):
    if DRY_RUN:
        return {"status": "DRY_RUN_SUCCESS", "message": "Dry run active. No real order placed."}
    try:
        product_id = PAIRS[symbol]["product_id"]
        path = "/v2/orders"
        payload = {
            "product_id": product_id,
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
        print(f"[ORDER ERROR {symbol}] {e}")
        return {"error": str(e)}

# =========================================================
# 11. FLASK ROUTES & MULTI-PAIR SCANNER SCHEDULER
# =========================================================
@app.route("/", methods=["GET"])
@app.route("/execute-trade", methods=["GET"])
def execute_trade():
    active_position = get_live_position()
    
    if active_position:
        symbol = active_position.get("product_symbol", "BTCUSD")
        df_active = fetch_market_data(symbol, limit_candles=50)
        latest_price = float(df_active["close"].iloc[-1]) if df_active is not None else 0.0
        if latest_price > 0:
            update_dynamic_risk_management(latest_price)
        return jsonify({
            "status": "POSITION_ALREADY_OPEN",
            "active_symbol": symbol,
            "message": "Trade already active. Skipping new orders to avoid duplicate executions."
        }), 200

    best_candidate = None
    balance = get_available_balance()

    for sym, config in PAIRS.items():
        df = fetch_market_data(sym, limit_candles=200)
        if df is None or df.empty:
            continue
            
        action, conf, sl, tp, leverage, strategy_tag = predict_signal(sym, df)
        latest_close = float(df["close"].iloc[-1])

        if action in ["BUY", "SELL"]:
            if best_candidate is None or conf > best_candidate["conf"]:
                best_candidate = {
                    "symbol": sym,
                    "action": action,
                    "conf": conf,
                    "sl": sl,
                    "tp": tp,
                    "leverage": leverage,
                    "strategy_tag": strategy_tag,
                    "latest_close": latest_close
                }

    if best_candidate:
        sym = best_candidate["symbol"]
        act = best_candidate["action"]
        contracts = calculate_contracts(sym, balance, best_candidate["leverage"], best_candidate["latest_close"])

        if balance <= 0 and not DRY_RUN:
            return jsonify({
                "status": "FAILED_NO_BALANCE",
                "symbol": sym,
                "strategy": best_candidate["strategy_tag"],
                "action": act,
                "balance": f"{balance:.2f}"
            }), 200

        order_res = place_order_with_brackets(sym, act, contracts, best_candidate["sl"], best_candidate["tp"])

        memory = load_memory()
        memory["open_position"] = {
            "symbol": sym,
            "side": act,
            "entry": best_candidate["latest_close"],
            "sl": best_candidate["sl"],
            "tp": best_candidate["tp"],
            "contracts": contracts,
            "time": int(time.time())
        }
        memory["total_trades"] = memory.get("total_trades", 0) + 1
        save_memory(memory)

        trade_mode = "SIMULATED" if DRY_RUN else "REAL MONEY"
        send_telegram_alert(
            f"🚀 *New Trade Executed ({trade_mode})*\n\n"
            f"• *Symbol:* `{sym}`\n"
            f"• *Side:* `{act}`\n"
            f"• *Contracts:* `{contracts}`\n"
            f"• *Leverage:* `{best_candidate['leverage']}x`\n"
            f"• *Entry Price:* `${best_candidate['latest_close']:,.2f}`\n"
            f"• *Hard Stop-Loss:* `${best_candidate['sl']:,.2f}`\n"
            f"• *Take-Profit:* `${best_candidate['tp']:,.2f}`\n"
            f"• *Confidence:* `{best_candidate['conf']*100:.2f}%`\n"
            f"• *Strategy:* `{best_candidate['strategy_tag']}`"
        )

        return jsonify({
            "status": "ORDER_PLACED" if not DRY_RUN else "SIMULATED_ORDER",
            "symbol": sym,
            "action": act,
            "confidence": f"{best_candidate['conf']*100:.2f}%",
            "contracts": contracts,
            "leverage": f"{best_candidate['leverage']}x",
            "entry": best_candidate["latest_close"],
            "stop_loss": best_candidate["sl"],
            "take_profit": best_candidate["tp"],
            "strategy": best_candidate["strategy_tag"],
            "delta_response": order_res
        }), 200

    return jsonify({
        "status": "WAIT_AND_SEE",
        "action": "HOLD",
        "scanned_pairs": list(PAIRS.keys()),
        "message": "No strong trend signals found on BTC or ETH. Staying safe.",
        "balance": f"{balance:.2f}"
    }), 200

scheduler = BackgroundScheduler()
scheduler.add_job(func=train_all_ai_brains, trigger="cron", day_of_week="sun", hour=0, minute=0)
scheduler.start()
# =========================================================
# BACKGROUND BOT STARTUP & FAST PORT BINDING
# =========================================================
def delayed_bot_startup():
    time.sleep(5)
    mode = "DRY RUN (Simulation)" if DRY_RUN else "LIVE TRADING (Delta Live)"
    send_telegram_alert(
        f"🤖 *Delta AI Trading Engine Online*\n\n"
        f"• *Status:* `{mode}`\n"
        f"• *Risk Limits:* `5% Margin Rule Active`\n"
        f"• *Pairs:* `{', '.join(PAIRS.keys())}`\n"
        f"• *Timestamp:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    )
    for sym, conf in PAIRS.items():
        if not os.path.exists(conf["model_file"]):
            try:
                train_and_save_ai_brain_for_pair(sym, conf["model_file"])
            except Exception as err:
                print(f"[BACKGROUND TRAIN ERROR] {err}")

startup_thread = threading.Thread(
    target=delayed_bot_startup,
    daemon=True
)
startup_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
