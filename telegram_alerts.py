import os
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram_alert(message: str) -> None:
  """Sends live trade notifications to configured Telegram chat."""
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        "[TELEGRAM] Token or Chat ID not configured in environment variables."
    )
    return

  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TELEGRAM_CHAT_ID,
      "text": message,
      "parse_mode": "Markdown",
  }

  try:
    response = requests.post(url, json=payload, timeout=5)
    if response.status_code != 200:
      print(f"[TELEGRAM ALERT FAILED] {response.text}")
  except Exception as e:
    print(f"[TELEGRAM ERROR] {e}")
    
