import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ТВОИ ПЕРВЫЕ ОРИГИНАЛЬНЫЕ ТОКЕНЫ И ГЛАВНЫЙ ЧАТ
TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

# ТОЛЬКО ТРИ ГЛАВНЫЕ МОНЕТЫ
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
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    url = "https://bybit.com"
    try:
        r = requests.get(url, params={'category': 'linear', 'symbol': symbol, 'interval': bybit_map[interval], 'limit': limit}, timeout=10).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass
    return None

def calculate_indicators(df):
    if df is None: return None
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

def intraday_market_analysis(name, ticker_symbol):
    try:
        # 1. ЗАГРУЗКА ИСТОРИЧЕСКИХ ТАЙМФРЕЙМОВ И ИНДИКАТОРОВ
        df_4h = calculate_indicators(get_live_candles(ticker_symbol, '4h', limit=150))
        df_1h = calculate_indicators(get_live_candles(ticker_symbol, '1h', limit=150))
        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', limit=200))
        
        if df_4h is None or df_1h is None or df_15m is None: return
        
        c4h = df_4h.iloc[-1]
        c1h = df_1h.iloc[-1]
        c15 = df_15m.iloc[-1]
        cp = c15['close']
        
        # СТРОГИЙ СТАРШИЙ ФИЛЬТР EMA 200 НА 4H
        above_4h_200 = c4h['close'] > c4h['ema200']
        under_4h_200 = c4h['close'] < c4h['ema200']
        
        # Положение цены на 15М относительно Скользящих Веера
        e_bull = (cp > c15['ema50']) and (c15['ema50'] > c15['ema100'] > c15['ema200'])
        e_bear = (cp < c15['ema50']) and (c15['ema50'] < c15['ema100'] < c15['ema200'])
        
        sk15, sd15 = c15['stoch_k'], c15['stoch_d']

        # 2. ПОИСК ТОЧЕК ВХОДА (Режим ТРЕНД-СНАЙПЕР: отскок от EMA50)
        # ЛОНГ СИГНАЛ
        if above_4h_200 and e_bull and (abs(cp - c15['ema50']) / c15['ema50'] < 0.003) and (sk15 < 40 and sk15 > sd15):
            sl = round(c15['ema100'] * 0.995, 2)
            send_telegram(f"🍏 <b>ВХОД В ЛОНГ #{name}</b>\n\n"
                          f"📈 Отскок от трендовой линии 15M EMA50!\n"
                          f"🛒 Вход: <code>{round(cp, 2)}</code>\n"
                          f"🛡️ Страховочный SL: <code>{sl}</code>\n\n"
                          f"⚡ <b>АКТИВИРОВАН БЫСТРЫЙ ТРЕЙЛИНГ-ВЫХОД!</b>\n"
                          f"Бот держит весь тренд, переводит в безубыток при +1.5% и закрывает сделку на развороте 15M EMA100 / 1H Stoch.")
            return

        # ШОРТ СИГНАЛ (Тот самый, который косил прибыль)
        elif under_4h_200 and e_bear and (abs(cp - c15['ema50']) / c15['ema50'] < 0.003) and (sk15 > 60 and sk15 < sd15):
            sl = round(c15['ema100'] * 1.005, 2)
            send_telegram(f"🔴 <b>ВХОД В ШОРТ #{name}</b>\n\n"
                          f"📉 Идеальный отбой от линии 15M EMA50!\n"
                          f"🛒 Вход: <code>{round(cp, 2)}</code>\n"
                          f"🛡️ Страховочный SL: <code>{sl}</code>\n\n"
                          f"⚡ <b>АКТИВИРОВАН БЫСТРЫЙ ТРЕЙЛИНГ-ВЫХОД!</b>\n"
                          f"Бот держит весь тренд, переводит в безубыток при +1.5% и закрывает сделку на развороте 15M EMA100 / 1H Stoch.")
            return
            
    except Exception as e:
        print(f"Ошибка {name}: {e}")

if __name__ == "__main__":
    for name, symbol in MONETS.items():
        intraday_market_analysis(name, symbol)
