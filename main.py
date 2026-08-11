import time
import hmac
import hashlib
import json
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

API_KEY = "KjHXEE687nqKhybHjZXmVZDkhLIrt"
API_SECRET = "Ti4ogJpiLP4KjhTJI5zytLCEd1Xz25NnCHlEp9z2u7PRYxspZCN4XaNI9Eia"
BASE_URL = "https://cdn.testnet.delta.exchange"

def generate_signature(method, endpoint, payload, timestamp):
    message = method + timestamp + endpoint + payload
    return hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

def get_delta_candles():
    url = f"{BASE_URL}/v2/chart/candles?resolution=1h&symbol=BTCUSD"
    res = requests.get(url).json()
    if 'result' in res and res['result']:
        df = pd.DataFrame(res['result'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    return None

def place_order():
    endpoint = "/v2/orders"
    timestamp = str(int(time.time()))
    payload = json.dumps({
        "product_id": 27,
        "size": 1,
        "side": "buy",
        "order_type": "market_order"
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

print("Delta Demo Bot Starting...", flush=True)

while True:
    try:
        print("Checking AI signal...", flush=True)
        data = get_delta_candles()
        
        if data is not None and len(data) > 30:
            data['Return'] = data['Close'].pct_change()
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data.dropna(inplace=True)

            X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X[:-5], y[:-5])

            latest_data = X.tail(1)
            prediction = model.predict(latest_data)

            if prediction[0] == 1:
                print("AI Signal: BUY -> Placing Demo Order...", flush=True)
                res = place_order()
                print("Delta API Response:", res, flush=True)
            else:
                print("AI Signal: NO TRADE / HOLD", flush=True)
        else:
            print("Failed to fetch candles from Delta", flush=True)

    except Exception as e:
        print("Bot Error:", e, flush=True)

    time.sleep(300)
            
