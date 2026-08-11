import time
from delta_rest_client import DeltaRestClient
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ૧. Delta Exchange API Credentials
API_KEY = "mrESi76tOg5F92F3gC3Pw6Ls0dgQH"
API_SECRET = "5Mcgity4tkkZloa3U0CHf3aboJtVePHr7EMDK5EswFBJzuC7rBOzbsCSKKYk"

delta_client = DeltaRestClient(
    base_url='https://api.delta.exchange',
    api_key=API_KEY,
    api_secret=API_SECRET
)

# ૨. AI સિગ્નલ જનરેટ કરવું
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

# ૩. Delta પર ઓર્ડર મૂકવો
if prediction[0] == 1:
    print("AI Signal: BUY -> Executing Order on Delta Exchange...")
    try:
        response = delta_client.place_order(
            product_id=27,            # 27 = BTCUSD Futures
            size=1,                   # Order Quantity
            side='buy',
            order_type='market_order'
        )
        print("Delta Order Response:", response)
    except Exception as e:
        print("Order Error:", e)
else:
    print("AI Signal: NO TRADE / HOLD")
 
