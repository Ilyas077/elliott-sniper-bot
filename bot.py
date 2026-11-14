import os
import json
import time
import requests
import pandas as pd
import numpy as np

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())

MONETS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT', 'AVAX': 'AVAXUSDT', 'INJ': 'INJUSDT',
    'AAVE': 'AAVEUSDT', 'ARB': 'ARBUSDT', 'JUP': 'JUPUSDT',
    'LDO': 'LDOUSDT'
}

WINDOW_4H, WINDOW_15M, STATE_FILE = 4, 6, "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_trades": {}, "last_signals": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Не заданы TG_TOKEN или TG_CHAT_ID!")
        return
    url = f"https://telegram.org{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        print(f"TG status {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"Сбой TG: {e}")


def get_live_candles(symbol, interval, limit=250):
    # Зеркало 1: Binance Vision Data API
    try:
        r = requests.get(
            "https://binance.vision",
            params={'symbol': symbol, 'interval': interval, 'limit': limit},
            headers=HEADERS,
            timeout=8
        )
        if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 50:
            df = pd.DataFrame([row[:6] for row in r.json()], columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            for c in ['open', 'high', 'low', 'close']:
                df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass

    # Зеркало 2: Coinbase Exchange API
    coin = symbol.replace("USDT", "")
    gran_map = {'4h': 14400, '1h': 3600, '15m': 900}
    try:
        r2 = requests.get(
            f"https://coinbase.com{coin}-USD/candles",
            params={'granularity': gran_map[interval]},
            headers=HEADERS,
            timeout=8
        )
        if r2.status_code == 200 and isinstance(r2.json(), list) and len(r2.json()) > 50:
            df = pd.DataFrame(r2.json()[:limit], columns=['ts', 'low', 'high', 'open', 'close', 'vol'])
            df = df.iloc[::-1].reset_index(drop=True)
            for c in ['open', 'high', 'low', 'close']:
                df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='s') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass

    # Зеркало 3: Kraken Public API
    kr_map = {'4h': 240, '1h': 60, '15m': 15}
    kr_coin = "XBT" if coin == "BTC" else coin
    r3 = requests.get(
        "https://kraken.com",
        params={'pair': f"{kr_coin}USD", 'interval': kr_map[interval]},
        headers=HEADERS,
        timeout=8
    ).json()
    pair_key = [k for k in r3['result'].keys() if k != 'last']
    df = pd.DataFrame([row[:6] for row in r3['result'][pair_key][-limit:]], columns=['ts', 'open', 'high', 'low', 'close', 'vwap'])
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='s') + pd.Timedelta(hours=5)
    return df


def calculate_indicators(df):
    for span in (50, 100, 200):
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    low_14, high_14 = df['low'].rolling(14).min(), df['high'].rolling(14).max()
    df['stoch_k'] = (100 * (df['close'] - low_14) / (high_14 - low_14)).rolling(3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    return df


def get_wave_pivots(df, window):
    pivots = []
    for i in range(window, len(df) - window):
        c_date = df['datetime'].iloc[i].strftime('%d.%m %H:%M')
        if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
            pivots.append({'type': 'HIGH', 'price': df['high'].iloc[i], 'date': c_date})
        elif df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
            pivots.append({'type': 'LOW', 'price': df['low'].iloc[i], 'date': c_date})
    clean = []
    for p in pivots:
        if not clean or clean[-1]['type'] != p['type']:
            clean.append(p)
    return clean


def check_elliott_axioms(w, st='BULL'):
    if len(w) < 9:
        return False, "Мало точек"
    p0, p1, p2, p3, p4, p5, pa, pb, pc = w[-9:]
    if st == 'BULL' and p0['type'] == 'LOW':
        l1, l3, l5 = p1['price'] - p0['price'], p3['price'] - p2['price'], p5['price'] - p4['price']
        r1, r2, r3 = p2['price'] > p0['price'], not (l3 < l1 and l3 < l5), p4['price'] > p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] > p2['price'])
    elif st == 'BEAR' and p0['type'] == 'HIGH':
        l1, l3, l5 = p0['price'] - p1['price'], p2['price'] - p3['price'], p4['price'] - p5['price']
        r1, r2, r3 = p2['price'] < p0['price'], not (l3 < l1 and l3 < l5), p4['price'] < p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] < p2['price'])
    else:
        return False, "Смещение структуры"
    return (True, "ВАЛИДНА") if (r1 and r2 and (r3 or diag)) else (False, "Нарушены аксиомы")


def intraday_wave_analysis(name, ticker_symbol, active_trades, last_signals):
    try:
        df_4h = calculate_indicators(get_live_candles(ticker_symbol, '4h', 250))
        piv_4h = get_wave_pivots(df_4h, WINDOW_4H)
        c4h = df_4h.iloc[-1]
        above_4h = c4h['close'] > c4h['ema200']
        wave_4h = "LONG"
        if len(piv_4h) >= 4:
            vl = piv_4h[-1]
            if vl['type'] == 'LOW' and vl['price'] < piv_4h[-3]['price']:
                wave_4h = "SHORT"
            elif vl['type'] == 'HIGH' and vl['price'] < piv_4h[-2]['price']:
                wave_4h = "WAIT"
        trend_4h = "LONG" if (wave_4h == "LONG" and above_4h) else ("SHORT" if (wave_4h == "SHORT" and not above_4h) else "WAIT")

        df_1h = calculate_indicators(get_live_candles(ticker_symbol, '1h', 250))
        c1h, p1h = df_1h.iloc[-1], df_1h.iloc[-2]
        rsi_1h = round(c1h['rsi'], 1)
        sk_1h, sd_1h, psk_1h, psd_1h = c1h['stoch_k'], c1h['stoch_d'], p1h['stoch_k'], p1h['stoch_d']
        b_1h, s_1h = c1h['close'] > c1h['ema200'], c1h['close'] < c1h['ema200']

        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', 250))
        piv_15m = get_wave_pivots(df_15m, WINDOW_15M)
        curr, prev = df_15m.iloc[-1], df_15m.iloc[-2]
        cp, rsi_v = round(float(curr['close']), 4), round(curr['rsi'], 1)
        sk, sd, psk, psd = curr['stoch_k'], curr['stoch_d'], prev['stoch_k'], prev['stoch_d']
        stoch_up = (sk < 25) and (sk > sd) and (psk <= psd or psk < 25)
        stoch_dn = (sk > 75) and (sk < sd) and (psk >= psd or psk > 75)

        print(f"=== {name} | Price: {cp} | 4H: {trend_4h} | 15M RSI: {rsi_v} ===")
        summary_line = f"• <b>{name}</b> (${cp}): 4H {trend_4h} | 15M RSI {rsi_v}"

        pos = active_trades.get(name)
        if pos:
            side, entry_p, be_done, sl_p = pos['side'], pos['entry'], pos['be'], pos['sl']
            is_long = side == "LONG"
            pnl = round(((cp - entry_p) if is_long else (entry_p - cp)) / entry_p * 100, 2)
            if pnl >= 1.5 and not be_done:
                pos['be'], pos['sl'] = True, entry_p
                send_telegram(f"🛡️ <b>БЕЗУБЫТОК #{name} (+{pnl}%)</b>\nSL на вход: <code>{entry_p}</code>")
            if (is_long and cp <= sl_p) or (not is_long and cp >= sl_p):
                send_telegram(f"🛑 <b>STOP-LOSS #{name} ({side})</b>\nЦена: <code>{cp}</code> | Итог: {pnl}%")
                del active_trades[name]
                return summary_line
            exit_long = is_long and ((cp < curr['ema100']) or (sk_1h > 80 and sk_1h < sd_1h and psk_1h >= psd_1h))
            exit_short = (not is_long) and ((cp > curr['ema100']) or (sk_1h < 20 and sk_1h > sd_1h and psk_1h <= psd_1h))
            if exit_long or exit_short:
                send_telegram(f"💰 <b>ВЫХОД #{name} ({side})</b>\nЦена: <code>{cp}</code> | Итог: <b>{pnl}%</b>")
                del active_trades[name]
            return summary_line

        candle_ts = str(curr['ts'])
        if last_signals.get(name) == candle_ts:
            return summary_line

        if trend_4h == "LONG" and b_1h and rsi_1h < 68 and stoch_up and cp > curr['ema50']:
            ok_w, w_msg = check_elliott_axioms(piv_15m, 'BULL')
            sl = round(float(min(curr['ema200'], df_15m['low'].iloc[-5:].min())) * 0.998, 4)
            tp = round(cp + (cp - sl) * 2.5, 4)
            active_trades[name], last_signals[name] = {"side": "LONG", "entry": cp, "sl": sl, "be": False}, candle_ts
            send_telegram(f"🚀 <b>LONG #{name}</b>\nВход: <code>{cp}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\nRSI: {rsi_1h}/{rsi_v} | Волны: {w_msg} ({'✅' if ok_w else '⚠️'})")

        elif trend_4h == "SHORT" and s_1h and rsi_1h > 32 and stoch_dn and cp < curr['ema50']:
            ok_w, w_msg = check_elliott_axioms(piv_15m, 'BEAR')
            sl = round(float(max(curr['ema200'], df_15m['high'].iloc[-5:].max())) * 1.002, 4)
            tp = round(cp - (sl - cp) * 2.5, 4)
