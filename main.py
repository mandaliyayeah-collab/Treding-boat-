import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

API_KEY = "KjHXEE687nqKhybHjZXmVZDkhLIrt"
API_SECRET = "Ti4ogJpiLP4KjhTJI5zytLCEd1Xz25NnCHlEp9z2u7PRYxspZCN4XaNI9Eia"
BASE_URL = "https://cdn.testnet.delta.exchange"

def generate_signature(method, endpoint, payload, timestamp):
    message = method + timestamp + endpoint + payload
    return hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_delta_candles():
    # Fetching 1-hour candles for stable market view
    url = f"{BASE_URL}/v2/chart/candles?resolution=1h&symbol=BTCUSD"
    res = requests.get(url).json()
    if 'result' in res and res['result']:
        df = pd.DataFrame(res['result'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['RSI'] = calculate_rsi(df)
        return df
    return None

def place_order_with_risk_management(current_price):
    endpoint = "/v2/orders"
    timestamp = str(int(time.time()))
    
    # Risk Management Settings: Stop-loss 1.5%, Take-profit 3.0%
    stop_loss = round(current_price * 0.985, 2)
    take_profit = round(current_price * 1.03, 2)
    
    payload = json.dumps({
        "product_id": 27,
        "size": 1,
        "side": "buy",
        "order_type": "market_order",
        "stop_loss_price": str(stop_loss),
        "take_profit_price": str(take_profit)
    })
    
    signature = generate_signature("POST", endpoint, payload, timestamp)
    headers = {
        "api-key": API_KEY,
        "signature": signature,
        "timestamp": timestamp,
        "Content-Type": "application/json"
    }
    
    response = requests.post(BASE_URL + endpoint, data=payload, headers=headers)
    return response.json()

print("Delta Advanced Bot Starting (30-Min Interval with Risk Management)...", flush=True)

while True:
    try:
        print("Fetching market data and calculating RSI/AI signals...", flush=True)
        data = get_delta_candles()
        
        if data is not None and len(data) > 30:
            data['Return'] = data['Close'].pct_change()
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data.dropna(inplace=True)

            X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'RSI']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X[:-5], y[:-5])

            latest_data = X.tail(1)
            prediction = model.predict(latest_data)
            current_rsi = latest_data['RSI'].values[0]
            current_price = latest_data['Close'].values[0]

            print(f"Current Price: {current_price} | RSI: {current_rsi:.2f}", flush=True)

            # Smart Filtering: Buy only if AI says BUY and RSI is not overbought (< 70)
            if prediction[0] == 1 and current_rsi < 70:
                print("AI Signal: HIGH PROBABILITY BUY -> Placing Order with Stop-Loss & Take-Profit...", flush=True)
                res = place_order_with_risk_management(current_price)
                print("Delta API Response:", res, flush=True)
            else:
                print("AI Signal: NO TRADE / HOLD (Market is risky or neutral)", flush=True)
        else:
            print("Failed to fetch candle data from Delta", flush=True)

    except Exception as e:
        print("Bot Error:", e, flush=True)

    # 30 Minutes Sleep Loop (1800 Seconds)
    print("Sleeping for 30 minutes until next scan...", flush=True)
    time.sleep(1800)
    
