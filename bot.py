import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime

# ТВОЙ НОВЫЙ ЧИСТЫЙ ТОКЕН ИЗ TELEGRAM
TG_TOKEN = "9255233645:AAEyzaiaJbfUxSq0W0-E3jsi6ZHXig-F5h0"
TG_CHAT_ID = "531386055"

MONETS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT'}
WINDOW_4H = 4   
WINDOW_15M = 6  

def send_telegram(message):
    url = f"https://telegram.org{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Ошибка TG: {e}")

def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '15m': '15'}
    url = "https://bybit.com"
    try:
        r = requests.get(url, params={'category': 'linear', 'symbol': symbol, 'interval': bybit_map[interval], 'limit': limit}, timeout=10).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            return df
    except Exception:
        pass
    return None

def calculate_indicators(df):
    if df is None: return None
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    fast_k = 100 * (df['close'] - low_14) / (high_14 - low_14 + 0.00001)
    df['stoch_k'] = fast_k.rolling(window=3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    return df

def get_wave_pivots(df, window):
    pivots = []
    for i in range(window, len(df) - window):
        is_high = df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max()
        is_low = df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min()
        if is_high: pivots.append({'type': 'HIGH', 'price': df['high'].iloc[i]})
        elif is_low: pivots.append({'type': 'LOW', 'price': df['low'].iloc[i]})
    clean = []
    for p in pivots:
        if not clean or clean[-1]['type'] != p['type']: clean.append(p)
    return clean

def check_elliott_axioms(w, st='BULL'):
    if len(w) < 6: return False
    p0, p1, p2, p3, p4, p5 = w[-6], w[-5], w[-4], w[-3], w[-2], w[-1]
    if st == 'BULL':
        w2_ok = p2['price'] > p0['price']
        w3_ok = (p3['price'] - p2['price']) > (p1['price'] - p0['price']) or (p3['price'] - p2['price']) > (p5['price'] - p4['price'])
        w4_ok = p4['price'] > p1['price']
    else:
        w2_ok = p2['price'] < p0['price']
        w3_ok = (p2['price'] - p3['price']) > (p0['price'] - p1['price']) or (p2['price'] - p3['price']) > (p4['price'] - p5['price'])
        w4_ok = p4['price'] < p1['price']
    return w2_ok and w3_ok and w4_ok

if __name__ == "__main__":
    for name, symbol in MONETS.items():
        df_4h = calculate_indicators(get_live_candles(symbol, '4h'))
        df_15m = calculate_indicators(get_live_candles(symbol, '15m'))
        if df_4h is None or df_15m is None: continue
        
        above_4h_ema = df_4h.iloc[-1]['close'] > df_4h.iloc[-1]['ema200']
        piv_15m = get_wave_pivots(df_15m, WINDOW_15M)
        if len(piv_15m) < 6: continue
        
        w15 = piv_15m[-6:]
        st_mode = 'BULL' if above_4h_ema else 'BEAR'
        
        if check_elliott_axioms(w15, st_mode):
            cp = df_15m.iloc[-1]['close']
            if st_mode == 'BULL':
                sl = round(w15[-1]['price'] * 0.995, 2)
                tp = round(cp * 1.015, 2)
                send_telegram(f"🍏 <b>СНАЙПЕР-ЛОНГ #{name} (Волна C)</b>\n\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl}</code>\n🎯 TP: <code>{tp}</code>")
            else:
                sl = round(w15[-1]['price'] * 1.005, 2)
                tp = round(cp * 0.985, 2)
                send_telegram(f"🍎 <b>СНАЙПЕР-ШОРТ #{name} (Волна C)</b>\n\n🛒 Вход: <code>{round(cp, 2)}</code>\n🛡️ SL: <code>{sl}</code>\n🎯 TP: <code>{tp}</code>")
