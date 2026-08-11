import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# 1. Fetch Stock / Index Data (e.g., NIFTY 50)
data = yf.download("^NSEI", start="2022-01-01", end="2026-01-01")
# 2. Features and Target Signal Creation
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]
y = data['Target']
# 3. Train AI Model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X[:-30], y[:-30]) # Train except last 30 days
# 4. Predict Signal for Latest Day
latest_data = X.tail(1)
prediction = model.predict(latest_data)
if prediction[0] == 1:
 print("AI Signal: BUY")
else:
 print("AI Signal: NO TRADE / HOLD")
