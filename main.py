import os
import time
import hmac
import hashlib
import json
import requests
import threading
from flask import Flask
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return "Delta Demo Bot is Active!"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

threading.Thread(target=run_web, daemon=True).start()

API_KEY = "KjHXEE687nqKhybHjZXmVZDkhLIrt"
API_SECRET = "Ti4ogJpiLP4KjhTJI5zytLCEd1Xz25NnCHlEp9z2u7PRYxspZCN4XaNI9Eia"
BASE_URL = "https://cdn.testnet.delta.exchange"

def generate_signature(method, endpoint, payload, timestamp):
    message = method + timestamp + endpoint + payload
    return hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()

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

def run_bot():
    print("Delta Demo Bot Starting...")
    while True:
        try:
            print("Checking AI signal...")
            data = yf.download("BTC-USD", start="2023-01-01", progress=False)
            data['Return'] = data['Close'].pct_change()
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data.dropna(inplace=True)

            X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X[:-30], y[:-30])

            latest_data = X.tail(1)
            prediction = model.predict(latest_data)

            if prediction[0] == 1:
                print("AI Signal: BUY -> Placing Demo Order...")
                res = place_order()
                print("Delta API Response:", res)
            else:
                print("AI Signal: NO TRADE / HOLD")
        except Exception as e:
            print("Bot Error:", e)

        time.sleep(300)

threading.Thread(target=run_bot, daemon=True).start()
