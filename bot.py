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
    "AVAXUSDT", "INJUSDT", "AAVEUSDT", "ARBUSDT", "JUPUSDT",
    "LDOUSDT", "SUIUSDT", "UNIUSDT", "TAOUSDT"
)
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()


def wait_for_candle_close():
    now = time.time()
    rem = 3600 - (now % 3600)
    if 0 < rem <= 150:
        print(f"Ждём закрытия 1H свечи: {round(rem, 1)} сек...")
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
        print("Ошибка: не заданы TG_TOKEN или TG_CHAT_ID")
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
        print(f"TG {r.status_code}: {r.text[:100]}")
        return r.status_code == 200
    except Exception as e:
        print(f"TG Err: {e}")
        return False


BINANCE_URL = base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
OKX_URL = base64.b64decode("aHR0cHM6Ly93d3cub2t4LmNvbS9hcGkvdjUvbWFya2V0L2NhbmRsZXM=").decode()


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
            for c in ("ts", "open", "high", "low", "close", "vol"):
                df.loc[:, c] = df.loc[:, c].astype(float)
            return df
    except Exception as e:
        print(f"Binance Vision err {symbol}: {e}")

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
            for c in ("ts", "open", "high", "low", "close", "vol"):
                df.loc[:, c] = df.loc[:, c].astype(float)
            return df
    except Exception as e:
        print(f"OKX err {symbol}: {e}")

    print(f"Не удалось загрузить {symbol} ({interval})")
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


def check_setup(cur, prv, t4h):
    v_ok = cur.get("vol") > cur.get("vol_sma") * 0.7
    if t4h == "LONG" and cur.get("ema9") > cur.get("ema21"):
        in_zone = (prv.get("low") <= prv.get("ema21") * 1.003) or (cur.get("low") <= cur.get("ema21") * 1.003)
        if in_zone and cur.get("close") > cur.get("ema50") and cur.get("close") > cur.get("ema21") and cur.get("close") > cur.get("open"):
            if (42 <= cur.get("rsi") <= 68) and v_ok:
                return "LONG", "PULLBACK V5"
    if t4h == "SHORT" and cur.get("ema9") < cur.get("ema21"):
        in_zone = (prv.get("high") >= prv.get("ema21") * 0.997) or (cur.get("high") >= cur.get("ema21") * 0.997)
        if in_zone and cur.get("close") < cur.get("ema50") and cur.get("close") < cur.get("ema21") and cur.get("close") < cur.get("open"):
            if (32 <= cur.get("rsi") <= 58) and v_ok:
                return "SHORT", "PULLBACK V5"
    return None, None


def make_chart(df, sym, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(55).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#131722")
    ax.set_facecolor("#131722")
    for i, r in sub.iterrows():
        col = "#089981" if r.get("close") >= r.get("open") else "#f23645"
        ax.plot((i, i), (r.get("low"), r.get("high")), color=col, lw=1)
        ax.bar(i, abs(r.get("close") - r.get("open")), bottom=min(r.get("open"), r.get("close")), width=0.6, color=col)
    ax.plot(sub.index, sub.loc[:, "ema9"], color="#2962ff", lw=1.2, label="EMA 9")
    ax.plot(sub.index, sub.loc[:, "ema21"], color="#ff9800", lw=1.2, label="EMA 21")
    ax.plot(sub.index, sub.loc[:, "ema50"], color="#ab47bc", lw=1.2, label="EMA 50")
    ax.axhline(entry, color="#ffffff", ls="--", lw=1, label=f"Entry {entry}")
    ax.axhline(sl, color="#f23645", lw=1.2, label=f"SL {sl}")
    ax.axhline(tp1, color="#089981", ls=":", lw=1.2, label=f"TP1 {tp1}")
    ax.axhline(tp2, color="#00e676", lw=1.2, label=f"TP2 {tp2}")
    ax.set_title(f"{sym} 1H / 4H | {side} ({setup})", color="white", fontsize=11)
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
    if val >= 100:
        return round(val, 2)
    elif val >= 1:
        return round(val, 4)
    else:
        return float(f"{val:.6g}")


def run_live():
    wait_for_candle_close()
    st = load_state()

    if not st.get("welcome_sent"):
        ok = send_tg(
            "👋 <b>Бот Signal Bot V5 запущен и подключён!</b>\n\n"
            "📊 Таймфреймы: <b>1H (вход) / 4H (тренд)</b>\n"
            f"🪙 Монет в наблюдении: <b>{len(SYMBOLS)}</b>\n"
            "🔔 Уведомления будут приходить автоматически при появлении сетапа."
        )
        if ok:
            st.update({"welcome_sent": True})
            save_state(st)

    df_btc = fetch_candles("BTCUSDT", "4h", 250)
    btc_t = trend_4h(add_ind(df_btc).iloc[-2]) if df_btc is not None else "FLAT"
    print(f"Старт проверки 1H/4H. Тренд BTC (4H): {btc_t}")

    sent_map = st.get("sent_signals", {})
    signals_found = 0

    for sym in SYMBOLS:
        df4 = fetch_candles(sym, "4h", 250)
        df1 = fetch_candles(sym, "1h", 200)
        if df4 is None or df1 is None:
            continue

        df4, df1 = add_ind(df4), add_ind(df1)
        r4, cur, prv = df4.iloc[-2], df1.iloc[-2], df1.iloc[-3]
        t4 = trend_4h(r4)

        print(f"{sym}: 4H={t4} | ADX={round(r4.get('adx'), 1)} | RSI(1H)={round(cur.get('rsi'), 1)}")

        key = f"{sym}_{int(cur.get('ts'))}"
        if key in sent_map:
            continue

        side, setup = check_setup(cur, prv, t4)
        if side and (sym == "BTCUSDT" or btc_t in (side, "FLAT")):
            signals_found += 1
            entry = fmt_price(cur.get("close"))
            risk = float(cur.get("atr")) * 2.0
            if side == "LONG":
                sl = fmt_price(entry - risk)
                tp1 = fmt_price(entry + risk)
                tp2 = fmt_price(entry + risk * 2)
            else:
                sl = fmt_price(entry + risk)
                tp1 = fmt_price(entry - risk)
                tp2 = fmt_price(entry - risk * 2)

            cap = (
                f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>SIGNAL BOT V5 — {side} ({setup})</b>\n\n"
                f"🪙 <b>#{sym}</b> (1H / 4H)\n"
                f"💵 Entry: <code>{entry}</code>\n"
                f"🛑 SL (2.0 ATR): <code>{sl}</code>\n"
                f"🎯 TP1 (50%, 1R): <code>{tp1}</code>\n"
                f"🎯 TP2 (50%, 2R): <code>{tp2}</code>\n"
                f"📊 RSI (1H): <b>{round(cur.get('rsi'), 1)}</b> | ADX (4H): <b>{round(r4.get('adx'), 1)}</b> | BTC: {btc_t}"
            )
            send_tg(cap, make_chart(df1, sym, side, setup, entry, sl, tp1, tp2))
            sent_map.update({key: side})

        time.sleep(0.25)

    keys = list(sent_map.keys())[-100:]
    st.update({"sent_signals": {k: sent_map.get(k) for k in keys}})
    save_state(st)
    print(f"Проверка завершена. Найдено сигналов: {signals_found}")


if __name__ == "__main__":
    run_live()
