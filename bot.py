import os, io, json, time, base64, requests
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
BYBIT_URL = "https://bybit.com"

def wait_for_candle_close():
    now = time.time()
    rem = 3600 - (now % 3600)
    if 0 < rem <= 120:
        print(f"Ждём закрытия 1H свечи: {round(rem, 1)} сек...")
        time.sleep(rem + 2.0)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "sent_signals" not in d: d["sent_signals"] = {}
                return d
        except Exception:
            pass
    return {"sent_signals": {}, "welcome_sent": False}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def send_tg(text, img_buf=None):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Ошибка: не заданы TG_TOKEN или TG_CHAT_ID")
        return
    try:
        if img_buf:
            r = requests.post(f"{TG_BASE}{TG_TOKEN}/sendPhoto", data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"}, files={"photo": ("chart.png", img_buf, "image/png")}, timeout=15)
        else:
            r = requests.post(f"{TG_BASE}{TG_TOKEN}/sendMessage", json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        print(f"TG {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"TG Err: {e}")

def fetch_candles(symbol, interval, limit=250):
    tf_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bybit_tf = tf_map.get(interval, "60")
    try:
        r = requests.get(BYBIT_URL, params={"category": "linear", "symbol": symbol, "interval": bybit_tf, "limit": limit}, headers=HEADERS, timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("retCode") == 0:
            rows = data["result"]["list"]
            if len(rows) > 50:
                df = pd.DataFrame([x[:6] for x in rows[::-1]], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["ts", "open", "high", "low", "close", "vol"]:
                    df[c] = df[c].astype(float)
                return df
        else:
            print(f"Bybit err {symbol}: {data.get('retMsg')}")
    except Exception as e:
        print(f"Ошибка запроса {symbol}: {e}")
    return None

def add_ind(df):
    for p in (9, 21, 50):
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    d = df["close"].diff()
    up_ema = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn_ema = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + up_ema / dn_ema))
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)).ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df

def trend_4h(r):
    if r["adx"] < 22: return "FLAT"
    if r["ema21"] > r["ema50"] and r["close"] > r["ema50"]: return "LONG"
    if r["ema21"] < r["ema50"] and r["close"] < r["ema50"]: return "SHORT"
    return "FLAT"

def check_setup(cur, prv, t4h):
    v_ok = cur["vol"] > cur["vol_sma"] * 1.1
    if t4h == "LONG" and cur["ema9"] > cur["ema21"]:
        in_zone = (prv["low"] <= prv["ema21"] * 1.003) or (cur["low"] <= cur["ema21"] * 1.003)
        if in_zone and cur["close"] > cur["ema50"] and cur["close"] > cur["ema21"] and cur["close"] > cur["open"]:
            if (42 <= cur["rsi"] <= 68) and v_ok: return "LONG", "PULLBACK V5"
    if t4h == "SHORT" and cur["ema9"] < cur["ema21"]:
        in_zone = (prv["high"] >= prv["ema21"] * 0.997) or (cur["high"] >= cur["ema21"] * 0.997)
        if in_zone and cur["close"] < cur["ema50"] and cur["close"] < cur["ema21"] and cur["close"] < cur["open"]:
            if (32 <= cur["rsi"] <= 58) and v_ok: return "SHORT", "PULLBACK V5"
    return None, None

def make_chart(df, sym, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(55).reset_index(drop=True)
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
    ax.set_title(f"{sym} 1H (Bybit) | {side} ({setup})", color="white", fontsize=11)
    ax.tick_params(colors="#b2b5be")
    ax.legend(facecolor="#1e222d", labelcolor="white", loc="upper left", fontsize=8)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def fmt_price(val):
    val = float(val)
    if val >= 100: return round(val, 2)
    elif val >= 1: return round(val, 4)
    else: return float(f"{val:.6g}")

def run_live():
    wait_for_candle_close()
    st = load_state()
    if not st.get("welcome_sent"):
        welcome_text = "🤖 <b>Signal Bot V5 запущен и подключен к Bybit!</b>\n\n⚙️ <b>Настройки:</b>\n• Таймфреймы: <b>4H</b> (тренд) + <b>1H</b> (вход)\n• Фильтр силы тренда: <b>ADX ≥ 22</b>\n• Риск-менеджмент: SL за локальный экстремум, TP1 <b>1.5R</b> / TP2 <b>3.0R</b>\n• Монет в мониторинге: <b>14</b>\n\nОжидаю закрытия свечей и формирования сетапов..."
        send_tg(welcome_text)
        st["welcome_sent"] = True

    df_btc_4h = fetch_candles("BTCUSDT", "4h", 250)
    if df_btc_4h is not None:
        df_btc_4h = add_ind(df_btc_4h)
        btc_t = trend_4h(df_btc_4h.iloc[-2])
    else:
        btc_t = "FLAT"

    print(f"Старт проверки (4H/1H Bybit). Тренд BTC (4H): {btc_t}")
    signals_found = 0

    for sym in SYMBOLS:
        df4 = df_btc_4h if sym == "BTCUSDT" else fetch_candles(sym, "4h", 250)
        df1 = fetch_candles(sym, "1h", 200)
        if df4 is None or df1 is None: continue
        if sym != "BTCUSDT": df4 = add_ind(df4)
        df1 = add_ind(df1)

        r4, cur, prv = df4.iloc[-2], df1.iloc[-2], df1.iloc[-3]
        t4 = trend_4h(r4)
        print(f"{sym}: 4H={t4} | ADX={round(r4['adx'], 1)} | RSI(1H)={round(cur['rsi'], 1)}")

        key = f"{sym}_{int(cur['ts'])}"
        if key in st["sent_signals"]: continue

        side, setup = check_setup(cur, prv, t4)
        if side and (sym == "BTCUSDT" or btc_t in (side, "FLAT")):
            entry = fmt_price(cur["close"])
            atr_buf = float(cur["atr"]) * 0.5
            recent_low = df1["low"].iloc[-7:-1].min()
            recent_high = df1["high"].iloc[-7:-1].max()
            if side == "LONG":
                sl = fmt_price(recent_low - atr_buf)
                risk = entry - sl
                if risk <= 0 or (risk / entry) > 0.045: continue
                tp1, tp2 = fmt_price(entry + risk * 1.5), fmt_price(entry + risk * 3.0)
            else:
                sl = fmt_price(recent_high + atr_buf)
                risk = sl - entry
                if risk <= 0 or (risk / entry) > 0.045: continue
                tp1, tp2 = fmt_price(entry - risk * 1.5), fmt_price(entry - risk * 3.0)

            risk_pct = round((risk / entry) * 100, 2)
            signals_found += 1
            icon = "🚀🟢" if side == "LONG" else "🔻🔴"
            cap = f"{icon} <b>SIGNAL BOT V5 — {side} ({setup})</b>\n\n🪙 <b>#{sym}</b> (1H / 4H Bybit)\n💵 Entry: <code>{entry}</code>\n🛑 SL (-{risk_pct}%): <code>{sl}</code>\n🎯 TP1 (50%, 1.5R): <code>{tp1}</code>\n🎯 TP2 (50%, 3.0R): <code>{tp2}</code>\n📊 RSI(1H): <b>{round(cur['rsi'], 1)}</b> | ADX(4H): <b>{round(r4['adx'], 1)}</b> | BTC: {btc_t}"
            send_tg(cap, make_chart(df1, sym, side, setup, entry, sl, tp1, tp2))
            st["sent_signals"][key] = int(time.time())
        time.sleep(0.2)

    now_ts = int(time.time())
    st["sent_signals"] = {k: v for k, v in st["sent_signals"].items() if isinstance(v, int) and (now_ts - v) < 172800}
    save_state(st)
    print(f"Проверка завершена. Найдено сигналов: {signals_found}")

run_live()
