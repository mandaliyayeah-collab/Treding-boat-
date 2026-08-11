import time
from delta_rest_client import DeltaRestClient
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ૧. Demo / Testnet API Credentials
API_KEY = "KjHXEE687nqKhybHjZXmVZDkhLIrt"
API_SECRET = "Ti4ogJpiLP4KjhTJI5zytLCEd1Xz25NnCHlEp9z2u7PRYxspZCN4XaNI9Eia"

# Demo Testnet URL કનેક્ટ કરી રહ્યા છીએ
delta_client = DeltaRestClient(
    base_url='https://cdn.testnet.delta.exchange',
    api_key=API_KEY,
    api_secret=API_SECRET
)

# ૨. AI મોડેલ ટ્રેનિંગ (BTC-USD)
data = yf.download("BTC-USD", start="2023-01-01")
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
y = data['Target']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X[:-30], y[:-30])

latest_data = X.tail(1)
prediction = model.predict(latest_data)

# ૩. Demo Delta પર ઓર્ડર એક્ઝિક્યુટ કરવો
if prediction[0] == 1:
    print("AI Signal: BUY -> Placing Order on Demo Delta Exchange...")
    try:
        response = delta_client.place_order(
            product_id=27,            # BTCUSD Futures
            size=1,                   # Demo Order Quantity
            side='buy',
            order_type='market_order'
        )
        print("Demo Order Response:", response)
    except Exception as e:
        print("Order Error:", e)
else:
    print("AI Signal: NO TRADE / HOLD")
    
