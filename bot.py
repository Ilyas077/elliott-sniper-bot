import os, io, sys, json, time, base64, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "INJUSDT", "AAVEUSDT", "ARBUSDT", "JUPUSDT", "LDOUSDT", "SUIUSDT", "UNIUSDT", "TAOUSDT"]
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
            d = json.load(open(STATE_FILE, "r", encoding="utf-8"))
            return d if "sent_signals" in d else {"sent_signals": {}}
        except Exception: pass
    return {"sent_signals": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, ensure_ascii=False, indent=2)

def send_tg(text, img_buf=None):
    if not TG_TOKEN or not TG_CHAT_ID: return
    try:
        if img_buf:
            r = requests.post(f"{TG_BASE}{TG_TOKEN}/sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"}, files={"photo": ("chart.png", img_buf, "image/png")}, timeout=15)
        else:
            r = requests.post(f"{TG_BASE}{TG_TOKEN}/sendMessage", json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        print(f"TG {r.status_code}: {r.text[:80]}")
    except Exception as e: print(f"TG Err: {e}")

def fetch_candles(symbol, interval, limit=250):
    for url in MIRRORS:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, headers=HEADERS, timeout=10)
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 20:
                df = pd.DataFrame([x[:6] for x in r.json()], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["open", "high", "low", "close", "vol"]: df[c] = df[c].astype(float)
                return df
        except Exception: pass
    return None

def add_ind(df):
    for p in (9, 21, 50, 200): df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    d = df["close"].diff()
    df["rsi"] = 100 - (100 / (1 + d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)))
    macd = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd_h"] = macd - macd.ewm(span=9, adjust=False).mean()
    h, l, c = df["high"], df["low"], df["close"]
    df["atr"] = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)).ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df

def trend_1h(r):
    if r["adx"] < 23: return "FLAT"
    if r["ema21"] > r["ema50"] > r["ema200"] and r["close"] > r["ema50"]: return "LONG"
    if r["ema21"] < r["ema50"] < r["ema200"] and r["close"] < r["ema50"]: return "SHORT"
    return "FLAT"

def check_setup(cur, prv, t1h):
    v_ok = cur["vol"] > cur["vol_sma"] * 0.9
    if t1h == "LONG" and cur["ema9"] > cur["ema21"] > cur["ema50"]:
        touched = prv["low"] <= prv["ema21"] * 1.001
        if touched and cur["close"] > cur["ema21"] and cur["close"] > cur["open"] and (52 <= cur["rsi"] <= 68) and cur["macd_h"] > 0 and v_ok:
            return "LONG", "PULLBACK V3"
    if t1h == "SHORT" and cur["ema9"] < cur["ema21"] < cur["ema50"]:
        touched = prv["high"] >= prv["ema21"] * 0.999
        if touched and cur["close"] < cur["ema21"] and cur["close"] < cur["open"] and (32 <= cur["rsi"] <= 48) and cur["macd_h"] < 0 and v_ok:
            return "SHORT", "PULLBACK V3"
    return None, None

def make_chart(df, sym, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#131722")
    ax.set_facecolor("#131722")
    for i, r in sub.iterrows():
        col = "#089981" if r["close"] >= r["open"] else "#f23645"
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1)
        ax.bar(i, abs(r["close"] - r["open"]), bottom=min(r["open"], r["close"]), width=0.6, color=col)
    ax.plot(sub.index, sub["ema9"], color="#2962ff", lw=1.2, label="EMA 9")
    ax.plot(sub.index, sub["ema21"], color="#ff9800", lw=1.2, label="EMA 21")
    ax.plot(sub.index, sub["ema50"], color="#ab47bc", lw=1.2, label="EMA 50")
    ax.axhline(entry, color="#ffffff", ls="--", lw=1, label=f"Entry {entry}")
    ax.axhline(sl, color="#f23645", lw=1.2, label=f"SL {sl}")
    ax.axhline(tp1, color="#089981", ls=":", lw=1.2, label=f"TP1 {tp1}")
    ax.axhline(tp2, color="#00e676", lw=1.2, label=f"TP2 {tp2}")
    ax.set_title(f"{sym} 15M | {side} ({setup})", color="white", fontsize=11)
    ax.tick_params(colors="#b2b5be")
    ax.legend(facecolor="#1e222d", labelcolor="white", loc="upper left", fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def run_live():
    st, summary = load_state(), []
    df_btc = fetch_candles("BTCUSDT", "1h", 250)
    btc_t = trend_1h(add_ind(df_btc).iloc[-2]) if df_btc is not None else "FLAT"
    for sym in SYMBOLS:
        df1, df15 = fetch_candles(sym, "1h", 250), fetch_candles(sym, "15m", 150)
        if df1 is None or df15 is None: continue
        df1, df15 = add_ind(df1), add_ind(df15)
        r1, cur, prv = df1.iloc[-2], df15.iloc[-2], df15.iloc[-3]
        t1 = trend_1h(r1)
        lp = round(float(df15.iloc[-1]["close"]), 4)
        summary.append(f"• <b>{sym}</b> (${lp}): 1H {t1} | ADX {round(r1['adx'], 1)} | RSI {round(cur['rsi'], 1)}")
        key = f"{sym}_{int(cur['ts'])}"
        if key in st["sent_signals"]: continue
        side, setup = check_setup(cur, prv, t1)
        if side and (sym == "BTCUSDT" or btc_t in (side, "FLAT")):
            entry, risk = round(float(cur["close"]), 4), float(cur["atr"]) * 2.2
            sl, tp1, tp2 = (round(entry - risk, 4), round(entry + risk, 4), round(entry + risk * 2, 4)) if side == "LONG" else (round(entry + risk, 4), round(entry - risk, 4), round(entry - risk * 2, 4))
            cap = f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>SIGNAL BOT V3 — {side} ({setup})</b>\n\n🪙 <b>#{sym}</b> (15M / 1H)\n💵 Entry: <code>{entry}</code>\n🛑 SL (2.2 ATR): <code>{sl}</code>\n🎯 TP1 (50%, 1R): <code>{tp1}</code>\n🎯 TP2 (50%, 2R): <code>{tp2}</code>\n📊 RSI: <b>{round(cur['rsi'], 1)}</b> | ADX: <b>{round(r1['adx'], 1)}</b> | BTC: {btc_t}"
            send_tg(cap, make_chart(df15, sym, side, setup, entry, sl, tp1, tp2))
            st["sent_signals"][key] = side
        time.sleep(0.4)
    save_state(st)

def run_backtest():
    res = []
    for sym in SYMBOLS:
        df1, df15 = fetch_candles(sym, "1h", 1000), fetch_candles(sym, "15m", 1500)
        if df1 is None or df15 is None: continue
        df1, df15 = add_ind(df1), add_ind(df15)
        df1["h_ts"] = (df1["ts"] // 3600000) * 3600000
        h_map = df1.set_index("h_ts").to_dict("index")
        trade = None
        for i in range(50, len(df15) - 1):
            cur, prv = df15.iloc[i], df15.iloc[i - 1]
            if trade is not None:
                side, entry, sl, tp1, tp2, hit1 = trade
                if side == "LONG":
                    if cur["low"] <= sl: res.append(0.5 if hit1 else -1.0); trade = None
                    elif not hit1 and cur["high"] >= tp1: trade = (side, entry, entry, tp1, tp2, True)
                    elif hit1 and cur["high"] >= tp2: res.append(1.5); trade = None
                else:
                    if cur["high"] >= sl: res.append(0.5 if hit1 else -1.0); trade = None
                    elif not hit1 and cur["low"] <= tp1: trade = (side, entry, entry, tp1, tp2, True)
                    elif hit1 and cur["low"] <= tp2: res.append(1.5); trade = None
                continue
            hr = h_map.get(((int(cur["ts"]) // 3600000) - 1) * 3600000)
            if not hr: continue
            side, _ = check_setup(cur, prv, trend_1h(hr))
            if side:
                e, r = cur["close"], cur["atr"] * 2.2
                trade = (side, e, e - r, e + r, e + r * 2, False) if side == "LONG" else (side, e, e + r, e - r, e - r * 2, False)
    if not res: send_tg("📊 Бэктест V3: сделок не найдено."); return
    wins, losses = [x for x in res if x > 0], [abs(x) for x in res if x < 0]
    eq = np.cumsum(res)
    dd = np.max(np.maximum.accumulate(eq) - eq)
    send_tg(f"📊 <b>БЭКТЕСТ SIGNAL BOT V3 (PULLBACK + 2.2 ATR + БУ)</b>\n• Сделок: <b>{len(res)}</b> | Winrate (в плюс/БУ): <b>{round(len(wins)/len(res)*100, 1)}%</b>\n• Profit Factor: <b>{round(sum(wins)/max(sum(losses), 1e-9), 2)}</b>\n• Total R: <b>{round(sum(res), 2)}R</b> | Max DD: <b>-{round(dd, 2)}R</b>")

if __name__ == "__main__":
    if os.environ.get("BOT_MODE") == "backtest": run_backtest()
    else: run_live()
