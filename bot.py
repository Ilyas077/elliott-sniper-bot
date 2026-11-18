import os, io, json, time, base64, requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "INJUSDT", "AAVEUSDT", "ARBUSDT", "JUPUSDT",
    "LDOUSDT", "SUIUSDT", "UNIUSDT", "TAOUSDT"
]
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()
MIRRORS = [
    base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
]

def wait_for_candle_close():
    now = time.time()
    rem = 900 - (now % 900)
    if 0 < rem <= 120:
        print(f"Ждём закрытия свечи: {round(rem, 1)} сек...")
        time.sleep(rem + 1.5)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if "sent_signals" in d else {"sent_signals": {}}
        except Exception:
            pass
    return {"sent_signals": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def send_tg(text, img_buf=None):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Ошибка: не заданы TG_TOKEN или TG_CHAT_ID")
        return
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
        print(f"TG {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"TG Err: {e}")

def fetch_candles(symbol, interval, limit=250):
    for url in MIRRORS:
        try:
            r = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                headers=HEADERS,
                timeout=10
            )
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 20:
                df = pd.DataFrame([x[:6] for x in r.json()], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["open", "high", "low", "close", "vol"]:
                    df[c] = df[c].astype(float)
                return df
            else:
                print(f"API {url} вернул статус {r.status_code} для {symbol}")
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
    df["atr"] = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
    
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)).ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df

def trend_1h(r):
    # Снизили порог ADX до 14, чтобы ловить выход из накопления в самом начале
    if r["adx"] < 14:
        return "FLAT"
    if r["ema21"] > r["ema50"] and r["close"] > r["ema50"]:
        return "LONG"
    if r["ema21"] < r["ema50"] and r["close"] < r["ema50"]:
        return "SHORT"
    return "FLAT"

def check_setup(df15, t1h):
    cur = df15.iloc[-2]
    prv = df15.iloc[-3]
    recent = df15.iloc[-14:-2]  # Предыдущие 12 свечей (3 часа)
    
    v_ok = cur["vol"] > cur["vol_sma"] * 0.70
    v_impulse = cur["vol"] > cur["vol_sma"] * 1.25  # Повышенный объем для пробоя

    # --- ЛОНГ ---
    if t1h == "LONG":
        # Сценарий 1: Откат к EMA 21 (классика)
        if cur["close"] > cur["ema9"]:
            touched = (prv["low"] <= prv["ema21"] * 1.004) or (cur["low"] <= cur["ema21"] * 1.004)
            if touched and cur["close"] > cur["ema21"] and cur["close"] > cur["open"] and (43 <= cur["rsi"] <= 72) and v_ok:
                return "LONG", "PULLBACK"
        
        # Сценарий 2: Пробой боковика на объеме (ловит импульсы вроде входа по 73.0)
        box_high = recent["high"].max()
        if (cur["close"] > box_high) and (prv["close"] <= box_high) and (cur["close"] > cur["open"]):
            if (cur["ema9"] > cur["ema21"]) and (52 <= cur["rsi"] <= 74) and v_impulse:
                return "LONG", "IMPULSE BREAKOUT"

    # --- ШОРТ ---
    if t1h == "SHORT":
        # Сценарий 1: Откат к EMA 21
        if cur["close"] < cur["ema9"]:
            touched = (prv["high"] >= prv["ema21"] * 0.996) or (cur["high"] >= prv["ema21"] * 0.996)
            if touched and cur["close"] < cur["ema21"] and cur["close"] < cur["open"] and (28 <= cur["rsi"] <= 57) and v_ok:
                return "SHORT", "PULLBACK"
                
        # Сценарий 2: Пробой поддержки вниз на объеме
        box_low = recent["low"].min()
        if (cur["close"] < box_low) and (prv["close"] >= box_low) and (cur["close"] < cur["open"]):
            if (cur["ema9"] < cur["ema21"]) and (26 <= cur["rsi"] <= 48) and v_impulse:
                return "SHORT", "IMPULSE BREAKOUT"

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
    ax.axhline(tp1, color="#089981", ls=":", lw=1.2, label=f"TP1 (+2.5%) {tp1}")
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

def fmt_price(val):
    return round(float(val), 6 if val < 1 else 4)

def run_live():
    wait_for_candle_close()
    st = load_state()
    df_btc = fetch_candles("BTCUSDT", "1h", 250)
    btc_t = trend_1h(add_ind(df_btc).iloc[-2]) if df_btc is not None else "FLAT"
    print(f"Старт проверки. Тренд BTC (1H): {btc_t}")
    
    signals_found = 0
    for sym in SYMBOLS:
        df1 = fetch_candles(sym, "1h", 250)
        df15 = fetch_candles(sym, "15m", 150)
        if df1 is None or df15 is None:
            continue
            
        df1, df15 = add_ind(df1), add_ind(df15)
        r1, cur = df1.iloc[-2], df15.iloc[-2]
        t1 = trend_1h(r1)
        
        print(f"{sym}: 1H={t1} | ADX={round(r1['adx'], 1)} | RSI={round(cur['rsi'], 1)}")
        
        key = f"{sym}_{int(cur['ts'])}"
        if key in st["sent_signals"]:
            continue
            
        side, setup = check_setup(df15, t1)
        if side and (sym == "BTCUSDT" or btc_t in (side, "FLAT")):
            signals_found += 1
            entry = fmt_price(cur["close"])
            risk = float(cur["atr"]) * 2.0
            if side == "LONG":
                sl = fmt_price(entry - risk)
                tp1 = fmt_price(entry * 1.025)
                tp2 = fmt_price(entry + risk * 2.5)
            else:
                sl = fmt_price(entry + risk)
                tp1 = fmt_price(entry * 0.975)
                tp2 = fmt_price(entry - risk * 2.5)
                
            cap = (
                f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>SIGNAL BOT V4.2 — {sym}</b>\n"
                f"Направление: <b>{side}</b> ({setup})\n"
                f"Таймфрейм: <b>15M</b> (Тренд 1H: {t1})\n\n"
                f"Вход: <code>{entry}</code>\n"
                f"Стоп-лосс: <code>{sl}</code>\n"
                f"Тейк 1 (2.5%): <code>{tp1}</code>\n"
                f"Тейк 2 (дальше): <code>{tp2}</code>\n\n"
                f"RSI: <b>{round(cur['rsi'], 1)}</b> | ADX 1H: <b>{round(r1['adx'], 1)}</b>"
            )
            chart = make_chart(df15, sym, side, setup, entry, sl, tp1, tp2)
            send_tg(cap, chart)
            st["sent_signals"][key] = int(time.time())
            
    cutoff = int(time.time()) - 86400
    st["sent_signals"] = {k: v for k, v in st["sent_signals"].items() if v > cutoff}
    save_state(st)
    print(f"Проверка завершена. Найдено сигналов: {signals_found}")

if __name__ == "__main__":
    run_live()
