import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

MONETS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT'}
WINDOW_4H = 4   
WINDOW_15M = 6  

def send_telegram(message):
    p1 = "https:" + "//" + "api" + "." + "telegram" + "." + "org"
    url = p1 + "/" + "bot" + TG_TOKEN + "/" + "sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Ошибка TG: {e}")

def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    b_url = "https:" + "//" + "api" + "." + "bybit" + "." + "com/v5/market/kline"
    try:
        r = requests.get(b_url, params={'category': 'linear', 'symbol': symbol, 'interval': bybit_map[interval], 'limit': limit}, timeout=10).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass
    bn_url = "https:" + "//" + "data-api" + "." + "binance" + "." + "vision/api/v3/klines"
    r2 = requests.get(bn_url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=10).json()
    df = pd.DataFrame(r2, columns=['ts', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig'])
    for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
    df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
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
        is_high = df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max()
        is_low = df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min()
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
    if len(w) < 9: return False, "Мало точек", False
    
    # Исправленная распаковка волновой структуры Эллиотта
    p0, p1, p2, p3, p4, p5 = w[0], w[1], w[2], w[3], w[4], w[5]
    
    if st == 'BULL' and p0['type'] == 'LOW':
        l1, l3, l5 = p1['price'] - p0['price'], p3['price'] - p2['price'], p5['price'] - p4['price']
        r1, r2, r3 = p2['price'] > p0['price'], not (l3 < l1 and l3 < l5), p4['price'] > p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] > p2['price'])
    elif st == 'BEAR' and p0['type'] == 'HIGH':
        l1, l3, l5 = p0['price'] - p1['price'], p2['price'] - p3['price'], p4['price'] - p5['price']
        r1, r2, r3 = p2['price'] < p0['price'], not (l3 < l1 and l3 < l5), p4['price'] < p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] < p2['price'])
    else:
        return False, "Смещение структуры", False
    if r1 and r2 and (r3 or diag):
        return True, "ВАЛИДНА", diag
    return False, "Нарушены аксиомы", diag

def intraday_wave_analysis(name, ticker_symbol):
    try:
        df_1h = get_live_candles(ticker_symbol, '1h', limit=300)
        high_low = df_1h['high'] - df_1h['low']
        high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
        low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
        df_1h['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        df_1h = calculate_indicators(df_1h)
        c1h = df_1h.iloc[-1]
        p1h = df_1h.iloc[-2]
        e50_1h, e100_1h, e200_1h = c1h['ema50'], c1h['ema100'], c1h['ema200']
        b_1h, s_1h = c1h['close'] > e200_1h, c1h['close'] < e200_1h
        
        atr_limits = {'BTC': 180.0, 'ETH': 9.0, 'SOL': 0.45}
        if c1h['atr'] < atr_limits.get(name, 0.0):
            print(f"   ⏸️ {name}: Флэт (ATR мал). Новые сигналы заблокированы.")
            return

        df_4h = calculate_indicators(get_live_candles(ticker_symbol, '4h', limit=250))
        piv_4h = get_wave_pivots(df_4h, WINDOW_4H)
        above_4h_ema = df_4h.iloc[-1]['close'] > df_4h.iloc[-1]['ema200']
        
        trend_4h = "LONG"
        if len(piv_4h) >= 4:
            vl = piv_4h[-1]
            if vl['type'] == 'LOW' and vl['price'] < piv_4h[-3]['price']: trend_4h = "SHORT"
            elif vl['type'] == 'HIGH' and vl['price'] < piv_4h[-2]['price']: trend_4h = "WAIT"
        
        if not ((trend_4h == "LONG" and above_4h_ema) or (trend_4h == "SHORT" and not above_4h_ema)):
            trend_4h = "WAIT"

        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', limit=350))
        piv_15m = get_wave_pivots(df_15m, WINDOW_15M)
        curr, prev = df_15m.iloc[-1], df_15m.iloc[-2]
        cp, rsi_v = curr['close'], round(curr['rsi'], 1)
        sk, sd = round(curr['stoch_k'], 1), round(curr['stoch_d'], 1)
        psk, psd = prev['stoch_k'], prev['stoch_d']
        
        stoch_cross_up = (sk < 25) and (sk > sd) and (psk <= psd or psk < 25)
        stoch_cross_down = (sk > 75) and (sk < sd) and (psk >= psd or psk > 75)
        
        candle_time_1h = c1h['datetime'].strftime('%d.%m %H:%M')
        if p1h['ema50'] <= p1h['ema100'] and c1h['ema50'] > c1h['ema100']:
            send_telegram(f"🌟 <b>КРЕСТ ТРЕНДА ВВЕРХ (1H) #{name}</b>\n\n📈 Быстрая EMA 50 пересекла EMA 100 снизу вверх!\n📍 Цена: <code>{round(cp, 2)}</code> USD")
        elif p1h['ema50'] >= p1h['ema100'] and c1h['ema50'] < c1h['ema100']:
            send_telegram(f"💀 <b>КРЕСТ ТРЕНДА ВНИЗ (1H) #{name}</b>\n\n📉 Быстрая EMA 50 пересекла EMA 100 сверху вниз!\n📍 Цена: <code>{round(cp, 2)}</code> USD")

        if b_1h and (cp > curr['ema50'] > curr['ema100'] > curr['ema200']) and (abs(cp - curr['ema50']) / curr['ema50'] < 0.003) and (sk < 40 and sk > sd) and trend_4h == "LONG":
            sl_t = round(curr['ema100'] * 0.995, 2)
            send_telegram(f"🚀 <b>ВХОД В ЛОНГ #{name} (Отскок EMA50)</b>\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl_t}</code>")
            return
        elif s_1h and (cp < curr['ema50'] < curr['ema100'] < curr['ema200']) and (abs(cp - curr['ema50']) / curr['ema50'] < 0.003) and (sk > 60 and sk < sd) and trend_4h == "SHORT":
            sl_t = round(curr['ema100'] * 1.005, 2)
            send_telegram(f"☄️ <b>ВХОД В ШОРТ #{name} (Отбой EMA50)</b>\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl_t}</code>")
            return

        if len(piv_15m) < 10: return
            
        if trend_4h == "LONG":
            w15 = piv_15m[-9:] if piv_15m[-1]['type'] == 'LOW' else piv_15m[-10:-1]
            ok, _, _ = check_elliott_axioms(w15, 'BULL')
            if ok and cp > w15[-1]['price'] and (cp - w15[-1]['price']) / w15[-1]['price'] < 0.006 and b_1h and rsi_v < 45 and stoch_cross_up and cp > curr['ema200']:
                sl = round(w15[-1]['price'] * 0.995, 2)
                send_telegram(f"🍏 <b>СНАЙПЕР-ЛОНГ #{name} (Волна C)</b>\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl}</code>")
        elif trend_4h == "SHORT":
            w15 = piv_15m[-9:] if piv_15m[-1]['type'] == 'HIGH' else piv_15m[-10:-1]
            ok, _, _ = check_elliott_axioms(w15, 'BEAR')
            if ok and cp < w15[-1]['price'] and (w15[-1]['price'] - cp) / w15[-1]['price'] < 0.006 and s_1h and rsi_v > 55 and stoch_cross_down and cp < curr['ema200']:
                sl = round(w15[-1]['price'] * 1.005, 2)
                send_telegram(f"🍎 <b>СНАЙПЕР-ШОРТ #{name} (Волна C)</b>\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl}</code>")
    except Exception as e:
        print(f"Ошибка {name}: {e}")

if __name__ == "__main__":
    for name, symbol in MONETS.items():
        intraday_wave_analysis(name, symbol)
