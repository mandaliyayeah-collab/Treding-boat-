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


app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_URL = os.environ.get(
    "DELTA_BASE_URL",
    "https://api.india.delta.exchange"
)

API_KEY = os.environ.get("DELTA_API_KEY", "")
API_SECRET = os.environ.get("DELTA_API_SECRET", "")

SYMBOL = "BTCUSD"
PRODUCT_ID = 27

MODEL_FILE = "ai_brain_model.pkl"
MEMORY_FILE = "trade_memory.json"

DRY_RUN = (
    os.environ.get("DRY_RUN", "true").lower() == "true"
)

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

    message = (
        method.upper()
        + timestamp
        + path
        + query
        + payload
    )

    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return signature, timestamp


def get_headers(method, path, query="", payload=""):
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Delta API credentials are missing."
        )

    signature, timestamp = generate_signature(
        API_SECRET,
        method,
        path,
        query,
        payload
    )

    return {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "BTC-AI-Trading-Bot/1.0"
    }


# =========================================================
# MEMORY
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
            data = json.load(f)

        base = default_memory()

        if isinstance(data, dict):
            base.update(data)

        return base

    except Exception as e:
        print(f"[MEMORY ERROR] {e}")
        return default_memory()


def save_memory(data):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(
                data,
                f,
                indent=4
            )

    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")


# =========================================================
# PRODUCT SPECS
# =========================================================

def get_product_specs():

    endpoints = [
        f"{BASE_URL}/v2/products/{SYMBOL}",
        f"https://api.india.delta.exchange/v2/products/{SYMBOL}",
        f"https://api.delta.exchange/v2/products/{SYMBOL}"
    ]

    for url in endpoints:
        try:
            response = requests.get(
                url,
                timeout=5,
                headers={"Accept": "application/json"}
            )

            data = response.json()

            if data.get("success") and data.get("result"):

                product = data.get("result")

                return {
                    "contract_value": float(
                        product.get(
                            "contract_value",
                            0.001
                        )
                    ),
                    "tick_size": float(
                        product.get(
                            "tick_size",
                            0.5
                        )
                    )
                }

        except Exception:
            continue

    return {
        "contract_value": 0.001,
        "tick_size": 0.5
    }


def round_to_tick(price, tick_size):

    if tick_size <= 0:
        return float(price)

    return round(
        round(price / tick_size) * tick_size,
        8
    )


# =========================================================
# WALLET BALANCE
# =========================================================

def get_available_balance():

    try:
        path = "/v2/wallet/balances"

        headers = get_headers(
            "GET",
            path
        )

        response = requests.get(
            BASE_URL + path,
            headers=headers,
            timeout=10
        )

        data = response.json()

        if not data.get("success"):
            return 0.0

        for item in data.get("result", []):

            if item.get("asset_symbol") in [
                "USD",
                "USDT",
                "INR"
            ]:

                balance = float(
                    item.get(
                        "available_balance",
                        0.0
                    )
                )

                if balance > 0:
                    return balance

        return 0.0

    except Exception as e:
        print(f"[BALANCE ERROR] {e}")
        return 0.0


# =========================================================
# MARKET DATA
# =========================================================

def fetch_market_data():

    try:
        now = int(time.time())

        lookback_seconds = (
            120 * 15 * 60
        )

        start_time = now - lookback_seconds

        base_urls = [
            BASE_URL.rstrip("/"),
            "https://api.india.delta.exchange",
            "https://api.delta.exchange"
        ]

        base_urls = list(
            dict.fromkeys(base_urls)
        )

        candles = None

        for base in base_urls:

            url = (
                f"{base}/v2/history/candles"
            )

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
                    timeout=10,
                    headers={
                        "Accept": "application/json"
                    }
                )

                if response.status_code != 200:
                    continue

                data = response.json()

                if (
                    data.get("result")
                    and len(data.get("result")) > 0
                ):

                    candles = data.get("result")
                    break

            except Exception:
                continue

        if not candles:
            print("[MARKET DATA] No candles received.")
            return None

        df = pd.DataFrame(candles)

        rename_map = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        }

        df = df.rename(
            columns=rename_map
        )

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in required_columns:

            if col not in df.columns:
                print(
                    f"[MARKET DATA] Missing column: {col}"
                )
                return None

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=required_columns
        ).copy()

        if "time" in df.columns:

            df["time"] = pd.to_numeric(
                df["time"],
                errors="coerce"
            )

            df = (
                df.dropna(subset=["time"])
                .sort_values("time")
                .drop_duplicates(
                    subset=["time"],
                    keep="last"
                )
            )

        return df.reset_index(
            drop=True
        )

    except Exception as e:

        print(f"[FETCH ERROR] {e}")
        return None


# =========================================================
# FUNDING RATE
# =========================================================

def fetch_funding_rate():

    endpoints = [
        f"{BASE_URL}/v2/tickers/{SYMBOL}",
        f"https://api.india.delta.exchange/v2/tickers/{SYMBOL}",
        f"https://api.delta.exchange/v2/tickers/{SYMBOL}"
    ]

    for url in endpoints:

        try:

            response = requests.get(
                url,
                timeout=5,
                headers={
                    "Accept": "application/json"
                }
            )

            data = response.json()

            if data.get("success"):

                ticker = data.get(
                    "result",
                    {}
                )

                return float(
                    ticker.get(
                        "funding_rate",
                        0.0
                    )
                )

        except Exception:
            continue

    return 0.0


# =========================================================
# ORDER BOOK
# =========================================================

def fetch_order_book_metrics():

    endpoints = [
        f"{BASE_URL}/v2/l2orderbook/{SYMBOL}",
        f"https://api.india.delta.exchange/v2/l2orderbook/{SYMBOL}",
        f"https://api.delta.exchange/v2/l2orderbook/{SYMBOL}"
    ]

    for url in endpoints:

        try:

            response = requests.get(
                url,
                timeout=5,
                headers={
                    "Accept": "application/json"
                }
            )

            data = response.json()

            if not data.get("success"):
                continue

            book = data.get(
                "result",
                {}
            )

            bids = book.get(
                "buy",
                []
            )

            asks = book.get(
                "sell",
                []
            )

            if bids and asks:

                top_bids = sum(
                    float(
                        x.get("size", 0)
                    )
                    for x in bids[:10]
                )

                top_asks = sum(
                    float(
                        x.get("size", 0)
                    )
                    for x in asks[:10]
                )

                total = (
                    top_bids
                    + top_asks
                    + 1e-9
                )

                obi = (
                    top_bids
                    - top_asks
                ) / total

                return (
                    obi,
                    top_bids,
                    top_asks
                )

        except Exception:
            continue

    return 0.0, 0.0, 0.0


# =========================================================
# INDICATOR ENGINE
# =========================================================

def add_indicators(df):

    df = df.copy()

    df["ma7"] = (
        df["close"]
        .rolling(7)
        .mean()
    )

    df["ma25"] = (
        df["close"]
        .rolling(25)
        .mean()
    )

    df["return"] = (
        df["close"]
        .pct_change()
    )

    high_low = (
        df["high"] - df["low"]
    )

    high_close = (
        df["high"]
        - df["close"].shift()
    ).abs()

    low_close = (
        df["low"]
        - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        tr.rolling(14)
        .mean()
    )

    delta = df["close"].diff()

    gain = (
        delta.where(
            delta > 0,
            0
        )
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.where(
            delta < 0,
            0
        )
        .rolling(14)
        .mean()
    )

    rs = gain / (
        loss + 1e-9
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    candle_range = (
        df["high"]
        - df["low"]
    ).replace(
        0,
        1e-9
    )

    body_low = df[
        ["open", "close"]
    ].min(axis=1)

    body_high = df[
        ["open", "close"]
    ].max(axis=1)

    df["lower_wick_ratio"] = (
        body_low - df["low"]
    ) / candle_range

    df["upper_wick_ratio"] = (
        df["high"] - body_high
    ) / candle_range

    df["vol_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["vol_surge"] = (
        df["volume"]
        / (
            df["vol_ma20"]
            + 1e-9
        )
    )

    df["norm_atr"] = (
        df["atr"]
        / (
            df["close"]
            + 1e-9
        )
    )

    df["wick_skew"] = (
        df["lower_wick_ratio"]
        - df["upper_wick_ratio"]
    )

    df["dist_ma25"] = (
        df["close"]
        - df["ma25"]
    ) / (
        df["ma25"]
        + 1e-9
    )

    up_move = (
        df["high"]
        - df["high"].shift(1)
    )

    down_move = (
        df["low"].shift(1)
        - df["low"]
    )

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    tr_smooth = (
        tr.rolling(14)
        .sum()
    )

    plus_di = (
        100
        * (
            plus_dm.rolling(14).sum()
            / (
                tr_smooth
                + 1e-9
            )
        )
    )

    minus_di = (
        100
        * (
            minus_dm.rolling(14).sum()
            / (
                tr_smooth
                + 1e-9
            )
        )
    )

    dx = (
        100
        * (
            (plus_di - minus_di).abs()
            / (
                plus_di
                + minus_di
                + 1e-9
            )
        )
    )

    df["adx"] = (
        dx.rolling(14)
        .mean()
    )

    sum_tr = (
        tr.rolling(14)
        .sum()
    )

    max_high = (
        df["high"]
        .rolling(14)
        .max()
    )

    min_low = (
        df["low"]
        .rolling(14)
        .min()
    )

    df["chop_index"] = (
        100
        * (
            np.log10(
                sum_tr
                / (
                    max_high
                    - min_low
                    + 1e-9
                )
            )
            / np.log10(14)
        )
    )

    df["recent_low"] = (
        df["low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    df["recent_high"] = (
        df["high"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["target"] = np.where(
        df["close"].shift(-1)
        > df["close"],
        1,
        0
    )

    return df


# =========================================================
# INSTITUTIONAL SWEEP
# =========================================================

def check_institutional_sweep(df):

    if len(df) < 30:
        return None

    latest = df.iloc[-1]

    atr = float(
        latest["atr"]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    obi_ratio, _, _ = (
        fetch_order_book_metrics()
    )

    funding_rate = (
        fetch_funding_rate()
    )

    tick_size = (
        get_product_specs()["tick_size"]
    )

    if (
        latest["low"]
        < latest["recent_low"]
        and latest["close"]
        > latest["recent_low"]
    ):

        if (
            obi_ratio > 0.35
            and latest[
                "lower_wick_ratio"
            ] >= 0.40
            and latest[
                "vol_surge"
            ] >= 1.20
            and funding_rate < 0.04
        ):

            sl = round_to_tick(
                latest["low"]
                - 0.4 * atr,
                tick_size
            )

            tp = round_to_tick(
                latest["close"]
                + 3.0 * atr,
                tick_size
            )

            return (
                "BUY",
                0.94,
                sl,
                tp,
                "INSTITUTIONAL_LIQUIDITY_HUNT_BUY"
            )

    if (
        latest["high"]
        > latest["recent_high"]
        and latest["close"]
        < latest["recent_high"]
    ):

        if (
            obi_ratio < -0.35
            and latest[
                "upper_wick_ratio"
            ] >= 0.40
            and latest[
                "vol_surge"
            ] >= 1.20
            and funding_rate > -0.04
        ):

            sl = round_to_tick(
                latest["high"]
                + 0.4 * atr,
                tick_size
            )

            tp = round_to_tick(
                latest["close"]
                - 3.0 * atr,
                tick_size
            )

            return (
                "SELL",
                0.94,
                sl,
                tp,
                "INSTITUTIONAL_LIQUIDITY_HUNT_SELL"
            )

    return None


# =========================================================
# AI PIPELINE
# =========================================================

def build_ai_pipeline():

    return Pipeline([
        (
            "scaler",
            RobustScaler()
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.06,
                max_depth=3,
                subsample=0.85,
                random_state=42
            )
        )
    ])


def train_and_save_ai_brain():

    df = fetch_market_data()

    if (
        df is None
        or len(df) < 30
    ):
        return None

    df = add_indicators(df)

    df_clean = df.dropna(
        subset=FEATURES + ["target"]
    ).copy()

    if len(df_clean) < 25:
        return None

    X = df_clean[
        FEATURES
    ][:-1]

    y = df_clean[
        "target"
    ][:-1]

    if y.nunique() < 2:
        print(
            "[AI] Training skipped: "
            "only one target class available."
        )
        return None

    pipeline = build_ai_pipeline()

    pipeline.fit(
        X,
        y
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    print(
        "[AI] Model trained and saved."
    )

    return pipeline


def get_or_load_ai_brain():

    if os.path.exists(
        MODEL_FILE
    ):

        try:
            return joblib.load(
                MODEL_FILE
            )

        except Exception as e:
            print(
                f"[AI LOAD ERROR] {e}"
            )

    return train_and_save_ai_brain()


# =========================================================
# DYNAMIC RISK MANAGEMENT
# =========================================================

def update_dynamic_risk_management(
    current_price
):

    memory = load_memory()

    pos = memory.get(
        "open_position"
    )

    if not pos:
        return

    side = pos.get(
        "side"
    )

    entry = float(
        pos.get("entry", 0)
    )

    sl = float(
        pos.get("sl", 0)
    )

    r_unit = abs(
        entry - sl
    )

    if r_unit <= 0:
  
