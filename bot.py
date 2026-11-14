import os
import json
import time
import requests
import pandas as pd
import numpy as np

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

MONETS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT', 'AVAX': 'AVAXUSDT', 'INJ': 'INJUSDT',
    'AAVE': 'AAVEUSDT', 'ARB': 'ARBUSDT', 'JUP': 'JUPUSDT',
    'LDO': 'LDOUSDT'
}

WINDOW_4H = 4
WINDOW_15M = 6
STATE_FILE = "state.json"


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
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Ошибка TG: {e}")


def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    b_url = "https://bybit.com"
    b_params = {
        'category': 'linear', 'symbol': symbol,
        'interval': bybit_map[interval], 'limit': limit
    }
    try:
        r = requests.get(b_url, params=b_params, timeout=5).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            cols = ['ts', 'open', 'high', 'low', 'close', 'vol', 'turn']
            df = pd.DataFrame(rows, columns=cols)
            for c in ['open', 'high', 'low', 'close']:
                df[c] = df[c].astype(float)
            ts_ms = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df['datetime'] = ts_ms + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass

    bn_url = "https://binance.vision"
    bn_params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    r2 = requests.get(bn_url, params=bn_params, timeout=5).json()
    cols2 = ['ts', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig']
    df = pd.DataFrame(r2, columns=cols2)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    ts_ms = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df['datetime'] = ts_ms + pd.Timedelta(hours=5)
    return df


def calculate_indicators(df):
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))

    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    fast_k = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_k'] = fast_k.rolling(window=3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    return df


def get_wave_pivots(df, window):
    pivots = []
    for i in range(window, len(df) - window):
        h_win = df['high'].iloc[i - window:i + window + 1].max()
        l_win = df['low'].iloc[i - window:i + window + 1].min()
        is_high = df['high'].iloc[i] == h_win
        is_low = df['low'].iloc[i] == l_win
        c_date = df['datetime'].iloc[i].strftime('%d.%m %H:%M')
        if is_high:
            pivots.append({'type': 'HIGH', 'price': df['high'].iloc[i], 'date': c_date})
        elif is_low:
            pivots.append({'type': 'LOW', 'price': df['low'].iloc[i], 'date': c_date})
    clean = []
    for p in pivots:
        if not clean or clean[-1]['type'] != p['type']:
            clean.append(p)
    return clean


def check_elliott_axioms(w, st='BULL'):
    if len(w) < 9:
        return False, "Мало точек", False
    p0, p1, p2, p3, p4, p5, pa, pb, pc = w[-9:]
    if st == 'BULL' and p0['type'] == 'LOW':
        l1 = p1['price'] - p0['price']
        l3 = p3['price'] - p2['price']
        l5 = p5['price'] - p4['price']
        r1 = p2['price'] > p0['price']
        r2 = not (l3 < l1 and l3 < l5)
        r3 = p4['price'] > p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] > p2['price'])
    elif st == 'BEAR' and p0['type'] == 'HIGH':
        l1 = p0['price'] - p1['price']
        l3 = p2['price'] - p3['price']
        l5 = p4['price'] - p5['price']
        r1 = p2['price'] < p0['price']
        r2 = not (l3 < l1 and l3 < l5)
        r3 = p4['price'] < p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] < p2['price'])
    else:
        return False, "Смещение структуры", False
    if r1 and r2 and (r3 or diag):
        return True, "ВАЛИДНА", diag
    return False, "Нарушены аксиомы", diag


def intraday_wave_analysis(name, ticker_symbol, active_trades, last_signals):
    try:
        df_4h = calculate_indicators(get_live_candles(ticker_symbol, '4h', limit=250))
        piv_4h = get_wave_pivots(df_4h, WINDOW_4H)
        c4h = df_4h.iloc[-1]
        e200_4h = c4h['ema200']
        above_4h_ema = c4h['close'] > e200_4h

        wave_4h = "LONG"
        if len(piv_4h) >= 4:
            vl = piv_4h[-1]
            if vl['type'] == 'LOW' and vl['price'] < piv_4h[-3]['price']:
                wave_4h = "SHORT"
            elif vl['type'] == 'HIGH' and vl['price'] < piv_4h[-2]['price']:
                wave_4h = "WAIT"

        if wave_4h == "LONG" and above_4h_ema:
            trend_4h = "LONG"
        elif wave_4h == "SHORT" and not above_4h_ema:
            trend_4h = "SHORT"
        else:
            trend_4h = "WAIT"

        df_1h = calculate_indicators(get_live_candles(ticker_symbol, '1h', limit=300))
        c1h, p1h = df_1h.iloc[-1], df_1h.iloc[-2]
        e200_1h = c1h['ema200']
        rsi_1h = round(c1h['rsi'], 1)
        sk_1h, sd_1h = round(c1h['stoch_k'], 1), round(c1h['stoch_d'], 1)
        psk_1h, psd_1h = p1h['stoch_k'], p1h['stoch_d']
        b_1h = c1h['close'] > e200_1h
        s_1h = c1h['close'] < e200_1h

        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', limit=350))
        piv_15m = get_wave_pivots(df_15m, WINDOW_15M)
        curr, prev = df_15m.iloc[-1], df_15m.iloc[-2]
        cp = round(float(curr['close']), 4)
        rsi_v = round(curr['rsi'], 1)
        sk, sd = round(curr['stoch_k'], 1), round(curr['stoch_d'], 1)
        psk, psd = prev['stoch_k'], prev['stoch_d']

        stoch_cross_up = (sk < 25) and (sk > sd) and (psk <= psd or psk < 25)
        stoch_cross_down = (sk > 75) and (sk < sd) and (psk >= psd or psk > 75)

        print(f"=== {name} | Price: {cp} | 4H: {trend_4h} | 15M RSI: {rsi_v} ===")
        summary_line = f"• <b>{name}</b> (${cp}): 4H {trend_4h} | 15M RSI {rsi_v}"

        pos = active_trades.get(name)
        if pos:
            side, entry_p, be_done, sl_p = pos['side'], pos['entry'], pos['be'], pos['sl']
            if side == "LONG":
                profit_pct = round((cp - entry_p) / entry_p * 100, 2)
                if profit_pct >= 1.5 and not be_done:
                    pos['be'] = True
                    pos['sl'] = entry_p
                    msg = f"🛡️ <b>БЕЗУБЫТОК #{name} (+{profit_pct}%)</b>\nSL на вход: <code>{entry_p}</code>"
                    send_telegram(msg)
                if cp <= sl_p:
                    msg = f"🛑 <b>STOP-LOSS #{name} (LONG)</b>\nЦена: <code>{cp}</code> | Итог: {profit_pct}%"
                    send_telegram(msg)
                    del active_trades[name]
                    return summary_line
                if (cp < curr['ema100']) or (sk_1h > 80 and sk_1h < sd_1h and psk_1h >= psd_1h):
                    msg = f"💰 <b>ВЫХОД #{name} (LONG)</b>\nЦена: <code>{cp}</code> | Итог: <b>{profit_pct}%</b>"
                    send_telegram(msg)
                    del active_trades[name]
                    return summary_line

            elif side == "SHORT":
                profit_pct = round((entry_p - cp) / entry_p * 100, 2)
                if profit_pct >= 1.5 and not be_done:
                    pos['be'] = True
                    pos['sl'] = entry_p
                    msg = f"🛡️ <b>БЕЗУБЫТОК #{name} (+{profit_pct}%)</b>\nSL на вход: <code>{entry_p}</code>"
                    send_telegram(msg)
                if cp >= sl_p:
                    msg = f"🛑 <b>STOP-LOSS #{name} (SHORT)</b>\nЦена: <code>{cp}</code> | Итог: {profit_pct}%"
                    send_telegram(msg)
                    del active_trades[name]
                    return summary_line
                if (cp > curr['ema100']) or (sk_1h < 20 and sk_1h > sd_1h and psk_1h <= psd_1h):
                    msg = f"💰 <b>ВЫХОД #{name} (SHORT)</b>\nЦена: <code>{cp}</code> | Итог: <b>{profit_pct}%</b>"
                    send_telegram(msg)
                    del active_trades[name]
                    return summary_line
            return summary_line

        candle_ts = str(curr['ts'])
        if last_signals.get(name) == candle_ts:
            return summary_line

        if trend_4h == "LONG" and b_1h and rsi_1h < 68 and stoch_cross_up and cp > curr['ema50']:
            ok_wave, wave_msg, _ = check_elliott_axioms(piv_15m, st='BULL')
            low_min = min(curr['ema200'], df_15m['low'].iloc[-5:].min())
            sl_price = round(float(low_min) * 0.998, 4)
            tp_price = round(cp + (cp - sl_price) * 2.5, 4)

            active_trades[name] = {"side": "LONG", "entry": cp, "sl": sl_price, "be": False}
            last_signals[name] = candle_ts
            mark = "✅" if ok_wave else "⚠️"

            msg = (
