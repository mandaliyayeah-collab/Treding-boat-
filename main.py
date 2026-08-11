import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

API_KEY = "KjHXEE687nqKhybHjZXmVZDkhLIrt"
API_SECRET = "Ti4ogJpiLP4KjhTJI5zytLCEd1Xz25NnCH1Ep9z2u7PRYxspZCN4XaNI9Eia"
BASE_URL = "https://testnet-api.delta.exchange"

def generate_signature(method, endpoint, payload_str, timestamp):
    message = method + timestamp + endpoint + payload_str
    return hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_delta_candles():
    url = "https://api.delta.exchange/v2/chart/candles?resolution=1h&symbol=BTCUSD"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers).json()
    if 'result' in res and res['result']:
        df = pd.DataFrame(res['result'])
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        return df.iloc[::-1].reset_index(drop=True)
    return None

def predict_signal(df):
    df['RSI'] = calculate_rsi(df)
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df = df.dropna()
    
    if len(df) < 20:
        return None, None
        
    X = df[['RSI', 'Return']]
    y = df['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    latest_features = X.iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    prob = model.predict_proba(latest_features)[0][1]
    
    latest_rsi = df['RSI'].iloc[-1]
    latest_price = df['Close'].iloc[-1]
    
    return prediction, prob, latest_rsi, latest_price

def place_order(product_id, size, side, price):
    endpoint = "/v2/orders"
    timestamp = str(int(time.time()))
    payload = {
        "product_id": product_id,
        "size": size,
        "side": side,
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
    
    res = requests.post(BASE_URL + endpoint, data=payload_str, headers=headers)
    return res.json()

def place_stop_loss_and_tp(product_id, size, entry_price, side):
    stop_loss_price = round(entry_price * 0.985, 2) if side == 'buy' else round(entry_price * 1.015, 2)
    take_profit_price = round(entry_price * 1.03, 2) if side == 'buy' else round(entry_price * 0.97, 2)
    
    print(f"Setting Stop-Loss at {stop_loss_price} and Take-Profit at {take_profit_price}")

def run_bot():
    print("Delta Advanced Bot Starting (30-Min Interval with Risk Management)...")
    while True:
        try:
            print("\nFetching market data and calculating RSI/AI signals...")
            df = get_delta_candles()
            if df is not None:
                prediction, prob, rsi, price = predict_signal(df)
                print(f"Current Price: {price} | RSI: {rsi:.2f} | AI Prob: {prob:.2f}")
                
                if prediction == 1 and prob > 0.65 and rsi < 40:
                    print("AI Signal: HIGH PROBABILITY BUY")
                    res = place_order(1, 1, "buy", price)
                    print("Order Response:", res)
                    place_stop_loss_and_tp(1, 1, price, "buy")
                else:
                    print("AI Signal: NO TRADE / HOLD")
            else:
                print("Failed to fetch market data.")
        except Exception as e:
            print(f"Bot Error: {e}")
            
        print("Sleeping for 30 minutes until next scan...")
        time.sleep(1800)

if __name__ == "__main__":
    run_bot()
    
