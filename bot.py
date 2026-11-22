import os, io, json, time, base64, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())
SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LTCUSDT", "DASHUSDT", "AVAXUSDT", "UNIUSDT", "AAVEUSDT"
)
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()
BINANCE_URL = base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
OKX_URL = base64.b64decode("aHR0cHM6Ly93d3cub2t4LmNvbS9hcGkvdjUvbWFya2V0L2NhbmRsZXM=").decode()


def wait_for_candle_close():
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return
    now = time.time()
    rem = 3600 - (now % 3600)
    if 0 < rem <= 150:
        print(f"Ждём закрытия 1H свечи: {round(rem, 1)} сек...", flush=True)
        time.sleep(rem + 2.0)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "sent_signals" not in d:
                d.update({"sent_signals": {}})
            if "welcome_sent" not in d:
                d.update({"welcome_sent": False})
            return d
        except Exception:
            pass
    return {"sent_signals": {}, "welcome_sent": False}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_tg(text, img_buf=None):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Ошибка: не заданы TG_TOKEN или TG_CHAT_ID", flush=True)
        return False
    try:
        if img_buf:
            r = requests.post(
                f"{TG_BASE}{TG_TOKEN}/sendPhoto",
                data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"},
                files={"photo": ("chart.png", img_buf, "image/png")},
                timeout=15
            )
        else:
            r = requests.post(
                f"{TG_BASE}{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10
            )
        print(f"TG {r.status_code}: {r.text[:100]}", flush=True)
        return r.status_code == 200
    except Exception as e:
        print(f"TG Err: {e}", flush=True)
        return False


def fetch_candles(symbol, interval, limit=250):
    try:
        r = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            headers=HEADERS,
            timeout=10
        )
        raw = r.json()
        if r.status_code == 200 and isinstance(raw, list) and len(raw) > 50:
            rows = list(map(lambda x: x[:6], raw))
            df = pd.DataFrame(rows, columns=("ts", "open", "high", "low", "close", "vol"))
            return df.astype(float)
    except Exception as e:
        print(f"Binance Vision err {symbol}: {e}", flush=True)

    try:
        inst_id = symbol.replace("USDT", "-USDT")
        bar = "1H" if interval == "1h" else ("4H" if interval == "4h" else interval)
        r = requests.get(
            OKX_URL,
            params={"instId": inst_id, "bar": bar, "limit": min(limit, 300)},
            headers=HEADERS,
            timeout=10
        )
        data = r.json()
        items = data.get("data", [])
        if r.status_code == 200 and data.get("code") == "0" and len(items) > 50:
            rows = list(map(lambda x: x[:6], items[::-1]))
            df = pd.DataFrame(rows, columns=("ts", "open", "high", "low", "close", "vol"))
            return df.astype(float)
    except Exception as e:
        print(f"OKX err {symbol}: {e}", flush=True)

    print(f"Не удалось загрузить {symbol} ({interval})", flush=True)
    return None


def add_ind(df):
    for p in (9, 21, 50):
        df.loc[:, f"ema{p}"] = df.loc[:, "close"].ewm(span=p, adjust=False).mean()
    d = df.loc[:, "close"].diff()
    up_ema = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn_ema = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
    df.loc[:, "rsi"] = 100 - (100 / (1 + up_ema / dn_ema))

    h, l, c = df.loc[:, "high"], df.loc[:, "low"], df.loc[:, "close"]
    tr = pd.concat((h - l, (h - c.shift()).abs(), (l - c.shift()).abs()), axis=1).max(axis=1)
    df.loc[:, "atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df.loc[:, "atr"].replace(0, 1e-9)
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df.loc[:, "atr"].replace(0, 1e-9)
    df.loc[:, "adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)).ewm(alpha=1/14, adjust=False).mean()

    df.loc[:, "vol_sma"] = df.loc[:, "vol"].rolling(20).mean()
    return df


def trend_4h(r):
    if r.get("adx") < 20:
        return "FLAT"
    if r.get("ema21") > r.get("ema50") and r.get("close") > r.get("ema50"):
        return "LONG"
    if r.get("ema21") < r.get("ema50") and r.get("close") < r.get("ema50"):
        return "SHORT"
    return "FLAT"
if __name__ == "__main__":
    mode = (os.environ.get("BOT_MODE") or "live").strip().lower()
    if mode == "backtest":
        run_backtest()
    else:
        run_live()
