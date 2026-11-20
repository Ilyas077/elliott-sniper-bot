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
COOLDOWN_SEC = 14400
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TG_BASE = base64.b64decode("aHR0cHM6Ly9hcGkudGVsZWdyYW0ub3JnL2JvdA==").decode()
MIRRORS = [
    base64.b64decode("aHR0cHM6Ly9kYXRhLWFwaS5iaW5hbmNlLnZpc2lvbi9hcGkvdjMva2xpbmVz").decode()
]


def wait_for_candle_close():
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return
    now = time.time()
    rem = 900 - (now % 900)
    if 0 < rem <= 120:
        print(f"Ждём закрытия свечи: {round(rem, 1)} сек...", flush=True)
        time.sleep(rem + 1.5)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "sent_signals" in d:
                    now_ts = int(time.time())
                    d["sent_signals"] = {
                        k: v for k, v in d["sent_signals"].items()
                        if now_ts - v < 86400
                    }
                    return d
        except Exception:
            pass
    return {"sent_signals": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_tg(text, img_buf=None):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Ошибка: не заданы TG_TOKEN или TG_CHAT_ID", flush=True)
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
        print(f"TG {r.status_code}: {r.text[:100]}", flush=True)
    except Exception as e:
        print(f"TG Err: {e}", flush=True)


def fetch_candles(symbol, interval, limit=250):
    for url in MIRRORS:
        try:
            r = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                headers=HEADERS,
                timeout=10
            )
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 30:
                df = pd.DataFrame([x[:6] for x in r.json()], columns=["ts", "open", "high", "low", "close", "vol"])
                for c in ["open", "high", "low", "close", "vol"]:
                    df[c] = df[c].astype(float)
                return df
            else:
                print(f"API {url} вернул статус {r.status_code} для {symbol}", flush=True)
        except Exception as e:
            print(f"Ошибка запроса {symbol}: {e}", flush=True)
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

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_up"] = sma20 + (std20 * 2.2)
    df["bb_low"] = sma20 - (std20 * 2.2)

    df["high_24"] = df["high"].rolling(24).max()
    df["low_24"] = df["low"].rolling(24).min()
    df["vol_sma"] = df["vol"].rolling(20).mean()
    return df


def trend_1h(r):
    if r["adx"] < 18:
        return "FLAT"
    if r["ema21"] > r["ema50"] and r["close"] > r["ema50"]:
        return "LONG"
    if r["ema21"] < r["ema50"] and r["close"] < r["ema50"]:
        return "SHORT"
    return "FLAT"
def get_trade_advice(side, t1):
    if side == t1:
        return "✅ <b>ПО ТРЕНДУ 1H:</b> высокий потенциал, можно тянуть до TP2"
    elif t1 == "FLAT":
        return "⚖️ <b>ФЛЭТ 1H:</b> часть забрать на TP1, остаток в безубыток"
    else:
        return "⚠️ <b>КОНТРТРЕНД 1H:</b> быстрый отскок — основную часть лучше забирать на TP1"


def check_setup(df15, r1):
    cur = df15.iloc[-2]
    prv = df15.iloc[-3]
    recent = df15.iloc[-16:-3]

    v_spike = cur["vol"] > cur["vol_sma"] * 1.3
    total_range = max(cur["high"] - cur["low"], 1e-9)
    upper_wick = cur["high"] - max(cur["open"], cur["close"])
    lower_wick = min(cur["open"], cur["close"]) - cur["low"]

    is_high_extreme = (cur["high"] >= cur["high_24"]) and (cur["high"] > cur["bb_up"] or prv["high"] > prv["bb_up"])
    rsi_overbought = (cur["rsi"] > 68) or (prv["rsi"] > 72)
    bear_rejection = (upper_wick > total_range * 0.4) or (cur["close"] < prv["low"] and cur["close"] < cur["open"])

    if is_high_extreme and rsi_overbought and bear_rejection and v_spike and cur["close"] < cur["bb_up"]:
        has_div = (cur["high"] > recent["high"].max()) and (max(cur["rsi"], prv["rsi"]) < recent["rsi"].max())
        htf_extreme = r1["rsi"] > 70
        if has_div and htf_extreme:
            setup = "🔥 ДИВЕРГЕНЦИЯ НА ПИКЕ (15M + 1H)"
        elif has_div:
            setup = "⚡ МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ"
        else:
            setup = "EXTREME HIGH REVERSAL"
        return "SHORT", setup

    is_low_extreme = (cur["low"] <= cur["low_24"]) and (cur["low"] < cur["bb_low"] or prv["low"] < prv["bb_low"])
    rsi_oversold = (cur["rsi"] < 32) or (prv["rsi"] < 28)
    bull_rejection = (lower_wick > total_range * 0.4) or (cur["close"] > prv["high"] and cur["close"] > cur["open"])

    if is_low_extreme and rsi_oversold and bull_rejection and v_spike and cur["close"] > cur["bb_low"]:
        has_div = (cur["low"] < recent["low"].min()) and (min(cur["rsi"], prv["rsi"]) > recent["rsi"].min())
        htf_extreme = r1["rsi"] < 30
        if has_div and htf_extreme:
            setup = "🔥 ДИВЕРГЕНЦИЯ НА ДНЕ (15M + 1H)"
        elif has_div:
            setup = "⚡ БЫЧЬЯ ДИВЕРГЕНЦИЯ"
        else:
            setup = "EXTREME LOW REVERSAL"
        return "LONG", setup

    return None, None


def make_chart(df, sym, side, setup, entry, sl, tp1, tp2):
    sub = df.tail(50).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#131722")
    ax.set_facecolor("#131722")

    for i, r in sub.iterrows():
        col = "#089981" if r["close"] >= r["open"] else "#f23645"
        ax.plot([i, i], [r["low"], r["high"]], color=col, lw=1)
        ax.bar(i, abs(r["close"] - r["open"]), bottom=min(r["open"], r["close"]), width=0.6, color=col)

    ax.plot(sub.index, sub["bb_up"], color="#787b86", ls="--", lw=0.9, alpha=0.8, label="BB Upper/Lower")
    ax.plot(sub.index, sub["bb_low"], color="#787b86", ls="--", lw=0.9, alpha=0.8)
    ax.fill_between(sub.index, sub["bb_low"], sub["bb_up"], color="#787b86", alpha=0.07)
    ax.plot(sub.index, sub["ema21"], color="#ff9800", lw=1.2, label="EMA 21")

    ax.axhline(entry, color="#ffffff", ls="--", lw=1, label=f"Entry {entry}")
    ax.axhline(sl, color="#f23645", lw=1.2, label=f"SL {sl}")
    ax.axhline(tp1, color="#089981", ls=":", lw=1.2, label=f"TP1 {tp1}")
    ax.axhline(tp2, color="#00e676", lw=1.2, label=f"TP2 {tp2}")

    ax.set_title(f"{sym} 15M | {side} | {setup}", color="white", fontsize=10)
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
    now_ts = int(time.time())
    is_manual = (
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        or os.environ.get("BOT_MODE", "").lower() in ("test", "manual", "debug")
    )

    df_btc = fetch_candles("BTCUSDT", "1h", 250)
    btc_t = trend_1h(add_ind(df_btc).iloc[-2]) if df_btc is not None else "FLAT"
    print(f"Старт проверки. Тренд BTC (1H): {btc_t}", flush=True)

    found_signals = []
    summary_lines = []

    for sym in SYMBOLS:
        df1 = fetch_candles(sym, "1h", 250)
        df15 = fetch_candles(sym, "15m", 150)
        if df1 is None or df15 is None:
            continue

        df1, df15 = add_ind(df1), add_ind(df15)
        r1, cur, prv = df1.iloc[-2], df15.iloc[-2], df15.iloc[-3]
        t1 = trend_1h(r1)

        print(f"{sym}: 1H={t1} | RSI_1H={round(r1['rsi'], 1)} | RSI_15m={round(cur['rsi'], 1)}", flush=True)
        summary_lines.append(f"• <b>{sym.replace('USDT', '')}:</b> {fmt_price(cur['close'])} | RSI {round(cur['rsi'])}/{round(r1['rsi'])} ({t1})")

        last_sent = st["sent_signals"].get(sym, 0)
        if now_ts - last_sent < COOLDOWN_SEC:
            continue

        side, setup = check_setup(df15, r1)
        if side:
            entry = fmt_price(cur["close"])
            atr_buf = float(cur["atr"]) * 0.5

            if side == "LONG":
                sl = fmt_price(min(cur["low"], prv["low"]) - atr_buf)
                risk = max(entry - sl, float(cur["atr"]))
                tp1 = fmt_price(entry + risk * 1.5)
                tp2 = fmt_price(entry + risk * 3.0)
            else:
                sl = fmt_price(max(cur["high"], prv["high"]) + atr_buf)
                risk = max(sl - entry, float(cur["atr"]))
                tp1 = fmt_price(entry - risk * 1.5)
                tp2 = fmt_price(entry - risk * 3.0)

            advice = get_trade_advice(side, t1)

            cap = (
                f"{'🚀🟢' if side == 'LONG' else '🔻🔴'} <b>EXTREME BOT V5 — {sym} ({side})</b>\n\n"
                f"📌 <b>Сетап:</b> {setup}\n"
                f"🎯 <b>Вход:</b> <code>{entry}</code>\n"
                f"🛑 <b>Стоп-лосс:</b> <code>{sl}</code>\n"
                f"✅ <b>Тейк 1 (1.5R):</b> <code>{tp1}</code>\n"
                f"✅ <b>Тейк 2 (3.0R):</b> <code>{tp2}</code>\n\n"
                f"📊 <b>RSI 15m / 1H:</b> {round(cur['rsi'], 1)} / {round(r1['rsi'], 1)}\n"
                f"🧭 <b>1H Фон ({sym} / BTC):</b> {t1} / {btc_t}\n"
                f"💡 {advice}"
            )

            chart = make_chart(df15, sym, side, setup, entry, sl, tp1, tp2)
            send_tg(cap, chart)

            st["sent_signals"][sym] = now_ts
            save_state(st)
            found_signals.append((sym, side))
            time.sleep(1.2)

    shorts = [s for s, d in found_signals if d == "SHORT"]
    longs = [s for s, d in found_signals if d == "LONG"]

    if len(shorts) >= 3:
        send_tg(
            f"🚨 <b>МАССОВЫЙ РАЗВОРОТ ВНИЗ (SHORT)</b>\n"
            f"Сразу {len(shorts)} монет достигли пика: <b>{', '.join(shorts)}</b>.\n"
            f"Высокая вероятность общего слива рынка!"
        )
    elif len(longs) >= 3:
        send_tg(
            f"🚨 <b>МАССОВЫЙ ОТСКОК ВВЕРХ (LONG)</b>\n"
            f"Сразу {len(longs)} монет достигли дна: <b>{', '.join(longs)}</b>.\n"
            f"Высокая вероятность общего разворота вверх!"
        )

    if not found_signals:
        print("Экстремальных точек разворота не найдено.", flush=True)
        if is_manual:
            report = "\n".join(summary_lines)
            send_tg(
                f"✅ <b>Ручная проверка (Бот V5 на связи)</b>\n"
                f"Тренд BTC (1H): <b>{btc_t}</b> | Экстремумов сейчас нет.\n\n"
                f"<b>Текущий срез (Цена | RSI 15m/1H | 1H):</b>\n{report}"
            )


if __name__ == "__main__":
    run_live()
