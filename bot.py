import os, io, sys, json, time, base64, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()
MIRRORS = [
    base64.b64decode("aHR0cHM6Ly9mYXBpLmJpbmFuY2UuY29tL2ZhcGkvdjEva2xpbmVz").decode(),
    base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
]

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"sent_signals": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, ensure_ascii=False, indent=2)

def send_telegram_photo(caption, img_buf):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Нет TG_TOKEN или TG_CHAT_ID!")
        return
    try:
        r = requests.post(
            f"{TG_BASE}{TG_TOKEN}/sendPhoto",
            data={"chat_id": TG_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", img_buf, "image/png")},
            timeout=15
        )
        print(f"TG Photo {r.status_code}: {r.text[:100]}")
    except Exception as e: print(f"Ошибка TG Photo: {e}")

def send_telegram_text(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Нет TG_TOKEN или TG_CHAT_ID!")
        return
    try:
        r = requests.post(
            f"{TG_BASE}{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        print(f"TG Text {r.status_code}: {r.text[:100]}")
    except Exception as e: print(f"Ошибка TG Text: {e}")

def fetch_candles(symbol, interval, limit=250, start_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    for url in MIRRORS:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 10:
                df = pd.DataFrame([x[:6] for x in r.json()], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["open", "high", "low", "close", "vol"]: df[c] = df[c].astype(float)
                df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
                return df
        except Exception: pass
    return None

def add_indicators(df):
    for p in (9, 21, 50, 200):
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    d = df["close"].diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move, dn_move = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df

def get_1h_trend(row_1h):
    if row_1h["adx"] < 18: return "FLAT"
    if row_1h["ema50"] > row_1h["ema200"] and row_1h["close"] > row_1h["ema200"]: return "LONG"
    if row_1h["ema50"] < row_1h["ema200"] and row_1h["close"] < row_1h["ema200"]: return "SHORT"
    return "FLAT"

def check_setup(cur, prv, trend_1h):
    vol_ok = cur["vol"] > cur["vol_sma"]
    if trend_1h == "LONG" and prv["ema9"] <= prv["ema21"] and cur["ema9"] > cur["ema21"] and cur["rsi"] > 50 and cur["macd_hist"] > 0 and vol_ok:
        return "LONG", "BREAKOUT"
    if trend_1h == "SHORT" and prv["ema9"] >= prv["ema21"] and cur["ema9"] < cur["ema21"] and cur["rsi"] < 50 and cur["macd_hist"] < 0 and vol_ok:
        return "SHORT", "BREAKOUT"
    if trend_1h == "LONG" and cur["ema9"] > cur["ema21"]:
        touched = (prv["low"] <= prv["ema21"] * 1.001) and (prv["close"] >= prv["ema21"] * 0.997)
        if touched and cur["close"] > cur["ema21"] and cur["close"] > cur["open"] and cur["rsi"] > 50 and cur["macd_hist"] > 0 and vol_ok:
            return "LONG", "PULLBACK"
    if trend_1h == "SHORT" and cur["ema9"] < cur["ema21"]:
        touched = (prv["high"] >= prv["ema21"] * 0.999) and (prv["close"] <= prv["ema21"] * 1.003)
        if touched and cur["close"] < cur["ema21"] and cur["close"] < cur["open"] and cur["rsi"] < 50 and cur["macd_hist"] < 0 and vol_ok:
            return "SHORT", "PULLBACK"
    return None, None

def build_chart(df, symbol, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(55).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#131722")
    ax.set_facecolor("#131722")
    for i, r in sub.iterrows():
        color = "#089981" if r["close"] >= r["open"] else "#f23645"
        ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1)
        ax.bar(i, abs(r["close"] - r["open"]), bottom=min(r["open"], r["close"]), width=0.6, color=color)
    ax.plot(sub.index, sub["ema9"], color="#2962ff", linewidth=1.2, label="EMA 9")
    ax.plot(sub.index, sub["ema21"], color="#ff9800", linewidth=1.3, label="EMA 21")
    ax.plot(sub.index, sub["ema50"], color="#ab47bc", linewidth=1.3, label="EMA 50")
    ax.axhline(entry, color="#ffffff", linestyle="--", linewidth=1, label=f"Entry: {entry}")
    ax.axhline(sl, color="#f23645", linestyle="-", linewidth=1.2, label=f"SL: {sl}")
    ax.axhline(tp1, color="#089981", linestyle=":", linewidth=1.2, label=f"TP1 (1R): {tp1}")
    ax.axhline(tp2, color="#00e676", linestyle="-", linewidth=1.2, label=f"TP2 (2R): {tp2}")
    ax.set_title(f"{symbol} 15M | {side} ({setup})", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#b2b5be")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left", fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf

def run_live():
    state = load_state()
    summary = []
    for sym in SYMBOLS:
        df1h = fetch_candles(sym, "1h", 250)
        df15m = fetch_candles(sym, "15m", 150)
        if df1h is None or df15m is None:
            summary.append(f"• <b>{sym}</b>: нет данных с биржи")
            continue
        df1h = add_indicators(df1h)
        df15m = add_indicators(df15m)
        row_1h = df1h.iloc[-2]
        trend_1h = get_1h_trend(row_1h)
        cur, prv = df15m.iloc[-2], df15m.iloc[-3]
        live_p = round(float(df15m.iloc[-1]["close"]), 4)
        summary.append(f"• <b>{sym}</b> (${live_p}): 1H {trend_1h} | ADX {round(row_1h['adx'], 1)} | 15M RSI {round(cur['rsi'], 1)}")
        ts_key = f"{sym}_{int(cur['ts'])}"
        if ts_key in state["sent_signals"]: continue
        side, setup = check_setup(cur, prv, trend_1h)
        if side:
            entry = round(float(cur["close"]), 4)
            risk = float(cur["atr"]) * 1.5
            if side == "LONG":
                sl, tp1, tp2 = round(entry - risk, 4), round(entry + risk, 4), round(entry + risk * 2, 4)
            else:
                sl, tp1, tp2 = round(entry + risk, 4), round(entry - risk, 4), round(entry - risk * 2, 4)
            caption = (
                f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>SIGNAL BOT V2 — {side} ({setup})</b>\n\n"
                f"🪙 Пара: <b>#{sym}</b> (15M / 1H)\n"
                f"💵 Entry: <code>{entry}</code>\n"
                f"🛑 Stop Loss: <code>{sl}</code>\n"
                f"🎯 TP1 (50%): <code>{tp1}</code> (1R)\n"
                f"🎯 TP2 (50%): <code>{tp2}</code> (2R)\n\n"
                f"📊 RSI 15M: <b>{round(cur['rsi'], 1)}</b> | ADX 1H: <b>{round(row_1h['adx'], 1)}</b>"
            )
            chart = build_chart(df15m, sym, side, setup, entry, sl, tp1, tp2)
            send_telegram_photo(caption, chart)
            state["sent_signals"][ts_key] = side
        time.sleep(0.5)
    if len(state["sent_signals"]) > 100:
        keys = list(state["sent_signals"].keys())[-100:]
        state["sent_signals"] = {k: state["sent_signals"][k] for k in keys}
    save_state(state)
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram_text("📡 <b>SIGNAL BOT V2 — Тестовый запуск пройден!</b>\n\n" + "\n".join(summary))

def run_backtest(days=365):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    stats = {"BREAKOUT": [], "PULLBACK": []}
    for sym in SYMBOLS:
        all_1h, cur_s = [], start_ms - 250 * 3600 * 1000
        while cur_s < end_ms:
            chunk = fetch_candles(sym, "1h", 1000, cur_s)
            if chunk is None or chunk.empty: break
            all_1h.append(chunk)
            cur_s = int(chunk["ts"].iloc[-1]) + 3600 * 1000
            time.sleep(0.2)
        all_15m, cur_s = [], start_ms - 100 * 900 * 1000
        while cur_s < end_ms:
import os, io, sys, json, time, base64, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()
MIRRORS = [
    base64.b64decode("aHR0cHM6Ly9mYXBpLmJpbmFuY2UuY29tL2ZhcGkvdjEva2xpbmVz").decode(),
    base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
]

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"sent_signals": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, ensure_ascii=False, indent=2)

def send_telegram_photo(caption, img_buf):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Нет TG_TOKEN или TG_CHAT_ID!")
        return
    try:
        r = requests.post(
            f"{TG_BASE}{TG_TOKEN}/sendPhoto",
            data={"chat_id": TG_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", img_buf, "image/png")},
            timeout=15
        )
        print(f"TG Photo {r.status_code}: {r.text[:100]}")
    except Exception as e: print(f"Ошибка TG Photo: {e}")

def send_telegram_text(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Нет TG_TOKEN или TG_CHAT_ID!")
        return
    try:
        r = requests.post(
            f"{TG_BASE}{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        print(f"TG Text {r.status_code}: {r.text[:100]}")
    except Exception as e: print(f"Ошибка TG Text: {e}")

def fetch_candles(symbol, interval, limit=250, start_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    for url in MIRRORS:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 10:
                df = pd.DataFrame([x[:6] for x in r.json()], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["open", "high", "low", "close", "vol"]: df[c] = df[c].astype(float)
                df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
                return df
        except Exception: pass
    return None

def add_indicators(df):
    for p in (9, 21, 50, 200):
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    d = df["close"].diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move, dn_move = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df

def get_1h_trend(row_1h):
    if row_1h["adx"] < 18: return "FLAT"
    if row_1h["ema50"] > row_1h["ema200"] and row_1h["close"] > row_1h["ema200"]: return "LONG"
    if row_1h["ema50"] < row_1h["ema200"] and row_1h["close"] < row_1h["ema200"]: return "SHORT"
    return "FLAT"

def check_setup(cur, prv, trend_1h):
    vol_ok = cur["vol"] > cur["vol_sma"]
    if trend_1h == "LONG" and prv["ema9"] <= prv["ema21"] and cur["ema9"] > cur["ema21"] and cur["rsi"] > 50 and cur["macd_hist"] > 0 and vol_ok:
        return "LONG", "BREAKOUT"
    if trend_1h == "SHORT" and prv["ema9"] >= prv["ema21"] and cur["ema9"] < cur["ema21"] and cur["rsi"] < 50 and cur["macd_hist"] < 0 and vol_ok:
        return "SHORT", "BREAKOUT"
    if trend_1h == "LONG" and cur["ema9"] > cur["ema21"]:
        touched = (prv["low"] <= prv["ema21"] * 1.001) and (prv["close"] >= prv["ema21"] * 0.997)
        if touched and cur["close"] > cur["ema21"] and cur["close"] > cur["open"] and cur["rsi"] > 50 and cur["macd_hist"] > 0 and vol_ok:
            return "LONG", "PULLBACK"
    if trend_1h == "SHORT" and cur["ema9"] < cur["ema21"]:
        touched = (prv["high"] >= prv["ema21"] * 0.999) and (prv["close"] <= prv["ema21"] * 1.003)
        if touched and cur["close"] < cur["ema21"] and cur["close"] < cur["open"] and cur["rsi"] < 50 and cur["macd_hist"] < 0 and vol_ok:
            return "SHORT", "PULLBACK"
    return None, None

def build_chart(df, symbol, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(55).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#131722")
    ax.set_facecolor("#131722")
    for i, r in sub.iterrows():
        color = "#089981" if r["close"] >= r["open"] else "#f23645"
        ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1)
        ax.bar(i, abs(r["close"] - r["open"]), bottom=min(r["open"], r["close"]), width=0.6, color=color)
    ax.plot(sub.index, sub["ema9"], color="#2962ff", linewidth=1.2, label="EMA 9")
    ax.plot(sub.index, sub["ema21"], color="#ff9800", linewidth=1.3, label="EMA 21")
    ax.plot(sub.index, sub["ema50"], color="#ab47bc", linewidth=1.3, label="EMA 50")
    ax.axhline(entry, color="#ffffff", linestyle="--", linewidth=1, label=f"Entry: {entry}")
    ax.axhline(sl, color="#f23645", linestyle="-", linewidth=1.2, label=f"SL: {sl}")
    ax.axhline(tp1, color="#089981", linestyle=":", linewidth=1.2, label=f"TP1 (1R): {tp1}")
    ax.axhline(tp2, color="#00e676", linestyle="-", linewidth=1.2, label=f"TP2 (2R): {tp2}")
    ax.set_title(f"{symbol} 15M | {side} ({setup})", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#b2b5be")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left", fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf

def run_live():
    state = load_state()
    summary = []
    for sym in SYMBOLS:
        df1h = fetch_candles(sym, "1h", 250)
        df15m = fetch_candles(sym, "15m", 150)
        if df1h is None or df15m is None:
            summary.append(f"• <b>{sym}</b>: нет данных с биржи")
            continue
        df1h = add_indicators(df1h)
        df15m = add_indicators(df15m)
        row_1h = df1h.iloc[-2]
        trend_1h = get_1h_trend(row_1h)
        cur, prv = df15m.iloc[-2], df15m.iloc[-3]
        live_p = round(float(df15m.iloc[-1]["close"]), 4)
        summary.append(f"• <b>{sym}</b> (${live_p}): 1H {trend_1h} | ADX {round(row_1h['adx'], 1)} | 15M RSI {round(cur['rsi'], 1)}")
        ts_key = f"{sym}_{int(cur['ts'])}"
        if ts_key in state["sent_signals"]: continue
        side, setup = check_setup(cur, prv, trend_1h)
        if side:
            entry = round(float(cur["close"]), 4)
            risk = float(cur["atr"]) * 1.5
            if side == "LONG":
                sl, tp1, tp2 = round(entry - risk, 4), round(entry + risk, 4), round(entry + risk * 2, 4)
            else:
                sl, tp1, tp2 = round(entry + risk, 4), round(entry - risk, 4), round(entry - risk * 2, 4)
            caption = (
                f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>SIGNAL BOT V2 — {side} ({setup})</b>\n\n"
                f"🪙 Пара: <b>#{sym}</b> (15M / 1H)\n"
                f"💵 Entry: <code>{entry}</code>\n"
                f"🛑 Stop Loss: <code>{sl}</code>\n"
                f"🎯 TP1 (50%): <code>{tp1}</code> (1R)\n"
                f"🎯 TP2 (50%): <code>{tp2}</code> (2R)\n\n"
                f"📊 RSI 15M: <b>{round(cur['rsi'], 1)}</b> | ADX 1H: <b>{round(row_1h['adx'], 1)}</b>"
            )
            chart = build_chart(df15m, sym, side, setup, entry, sl, tp1, tp2)
            send_telegram_photo(caption, chart)
            state["sent_signals"][ts_key] = side
        time.sleep(0.5)
    if len(state["sent_signals"]) > 100:
        keys = list(state["sent_signals"].keys())[-100:]
        state["sent_signals"] = {k: state["sent_signals"][k] for k in keys}
    save_state(state)
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram_text("📡 <b>SIGNAL BOT V2 — Тестовый запуск пройден!</b>\n\n" + "\n".join(summary))

def run_backtest(days=365):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    stats = {"BREAKOUT": [], "PULLBACK": []}
    for sym in SYMBOLS:
        all_1h, cur_s = [], start_ms - 250 * 3600 * 1000
        while cur_s < end_ms:
            chunk = fetch_candles(sym, "1h", 1000, cur_s)
            if chunk is None or chunk.empty: break
            all_1h.append(chunk)
            cur_s = int(chunk["ts"].iloc[-1]) + 3600 * 1000
            time.sleep(0.2)
        all_15m, cur_s = [], start_ms - 100 * 900 * 1000
        while cur_s < end_ms:
