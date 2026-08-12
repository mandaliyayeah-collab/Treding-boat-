import os
import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://cdn-ind.testnet.deltaex.org"

API_KEY = os.getenv("DELTA_API_KEY")
API_SECRET = os.getenv("DELTA_API_SECRET")

SYMBOL = "BTCUSD"
TIMEFRAME = "1h"

# શરૂઆતમાં true રાખો.
# true = માત્ર signal બતાવશે, order નહીં મૂકે.
# false = actual Demo/Testnet order મૂકી શકે છે.
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

ORDER_SIZE = 1

BUY_PROBABILITY = 0.65
BUY_RSI_MAX = 40

STOP_LOSS_PERCENT = 0.015
TAKE_PROFIT_PERCENT = 0.03

SCAN_INTERVAL = 1800  # 30 minutes


# =========================================================
# BASIC CHECK
# =========================================================

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "DELTA_API_KEY અને DELTA_API_SECRET Render Environment Variablesમાં નાખો."
    )


# =========================================================
# SIGNATURE
# =========================================================

def generate_signature(method, timestamp, path, query_string="", body=""):
    message = (
        method.upper()
        + timestamp
        + path
        + query_string
        + body
    )

    return hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


# =========================================================
# HEADERS
# =========================================================

def auth_headers(method, path, query_string="", body=""):
    timestamp = str(int(time.time()))

    signature = generate_signature(
        method,
        timestamp,
        path,
        query_string,
        body
    )

    return {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "delta-ai-bot/1.0"
    }


# =========================================================
# PRODUCT
# =========================================================

def get_product():
    path = f"/v2/products/{SYMBOL}"

    response = requests.get(
        BASE_URL + path,
        headers={
            "Accept": "application/json",
            "User-Agent": "delta-ai-bot/1.0"
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"Product API error: {data}")

    product = data["result"]

    print(
        f"Product: {product.get('symbol')} | "
        f"Product ID: {product.get('id', product.get('product_id'))}"
    )

    return product


# =========================================================
# CANDLES
# =========================================================

def get_delta_candles():
    path = "/v2/history/candles"

    end = int(time.time())
    start = end - (200 * 60 * 60)

    params = {
        "resolution": TIMEFRAME,
        "symbol": SYMBOL,
        "start": start,
        "end": end
    }

    response = requests.get(
        BASE_URL + path,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "delta-ai-bot/1.0"
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"Candle API error: {data}")

    candles = data.get("result", [])

    if not candles:
        return None

    df = pd.DataFrame(candles)

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df = df[required]

    df.columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df = df.astype(float)

    df = df.dropna()

    return df.reset_index(drop=True)


# =========================================================
# RSI
# =========================================================

def calculate_rsi(data, window=14):

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================================================
# AI SIGNAL
# =========================================================

def predict_signal(df):

    df = df.copy()

    df["RSI"] = calculate_rsi(df)

    df["Return"] = df["Close"].pct_change()

    # Next candle positive return = BUY target
    df["Target"] = np.where(
        df["Return"].shift(-1) > 0,
        1,
        0
    )

    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        return None

    features = ["RSI", "Return"]

    # છેલ્લી row prediction માટે રાખીશું
    latest = df.iloc[[-1]]

    train = df.iloc[:-1]

    X_train = train[features]
    y_train = train["Target"]

    if y_train.nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    X_latest = latest[features]

    prediction = int(
        model.predict(X_latest)[0]
    )

    probability = float(
        model.predict_proba(X_latest)[0][1]
    )

    latest_rsi = float(
        latest["RSI"].iloc[0]
    )

    latest_price = float(
        latest["Close"].iloc[0]
    )

    return (
        prediction,
        probability,
        latest_rsi,
        latest_price
    )


# =========================================================
# POSITION
# =========================================================

def get_position(product_id):

    path = "/v2/positions"

    query_string = f"product_id={product_id}"

    headers = auth_headers(
        "GET",
        path,
        query_string,
        ""
    )

    response = requests.get(
        BASE_URL + path,
        params={
            "product_id": product_id
        },
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"Position API error: {data}")

    return data.get("result", {})


# =========================================================
# PLACE BUY ORDER WITH SL + TP
# =========================================================

def place_buy_order(product_id, price):

    stop_loss = round(
        price * (1 - STOP_LOSS_PERCENT),
        2
    )

    take_profit = round(
        price * (1 + TAKE_PROFIT_PERCENT),
        2
    )

    print("")
    print("========== ORDER ==========")
    print(f"Side        : BUY")
    print(f"Price       : {price}")
    print(f"Size        : {ORDER_SIZE}")
    print(f"Stop Loss   : {stop_loss}")
    print(f"Take Profit : {take_profit}")
    print(f"DRY RUN     : {DRY_RUN}")
    print("============================")
    print("")

    if DRY_RUN:
        print("DRY RUN ON -> Actual order મૂક્યો નથી.")
        return {
            "dry_run": True,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    path = "/v2/orders"

    payload = {
        "product_id": product_id,
        "size": ORDER_SIZE,
        "side": "buy",
        "order_type": "limit_order",
        "limit_price": str(price),

        "time_in_force": "gtc",

        "reduce_only": False,

        "bracket_stop_trigger_method": "last_traded_price",

        "bracket_stop_loss_price": str(stop_loss),

        "bracket_take_profit_price": str(take_profit),

        "client_order_id": f"ai_{int(time.time())}"
    }

    body = json.dumps(
        payload,
        separators=(",", ":")
    )

    headers = auth_headers(
        "POST",
        path,
        "",
        body
    )

    response = requests.post(
        BASE_URL + path,
        data=body,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    print("ORDER RESPONSE:")
    print(json.dumps(data, indent=2))

    return data


# =========================================================
# MAIN SCAN
# =========================================================

def scan_market():

    print("")
    print("=" * 60)
    print("BTC AI BOT SCAN")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    df = get_delta_candles()

    if df is None:
        print("Candles મળ્યા નથી.")
        return

    signal = predict_signal(df)

    if signal is None:
        print("AI માટે પૂરતો data નથી.")
        return

    prediction, probability, rsi, price = signal

    print(f"Current Price : {price}")
    print(f"RSI           : {rsi:.2f}")
    print(f"AI Probability: {probability:.2f}")
    print(f"Prediction    : {prediction}")

    # Product ID
    product = get_product()

    product_id = product.get("id")

    if product_id is None:
        product_id = product.get("product_id")

    if product_id is None:
        raise RuntimeError(
            f"Product ID મળ્યો નથી: {product}"
        )

    # Existing position check
    position = get_position(product_id)

    position_size = float(
        position.get("size", 0) or 0
    )

    print(f"Current Position: {position_size}")

    if position_size != 0:
        print("Already position open છે -> New BUY નહીં.")
        return

    # BUY condition
    if (
        prediction == 1
        and probability >= BUY_PROBABILITY
        and rsi < BUY_RSI_MAX
    ):

        print("")
        print("🔥 HIGH PROBABILITY BUY SIGNAL")
        print("")

        place_buy_order(
            product_id,
            price
        )

    else:

        print("NO TRADE / HOLD")


# =========================================================
# BOT LOOP
# =========================================================

def run_bot():

    print("")
    print("======================================")
    print(" Delta AI Trading Bot")
    print(" Environment: TESTNET / DEMO")
    print(f" Symbol: {SYMBOL}")
    print(f" DRY_RUN: {DRY_RUN}")
    print("======================================")

    while True:

        try:

            scan_market()

        except Exception as e:

            print("")
            print("BOT ERROR:")
            print(str(e))
            print("")

        print("")
        print("Next scan after 30 minutes...")
        print("")

        time.sleep(SCAN_INTERVAL)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    run_bot()
