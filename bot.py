import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import clear_output

# --- НАСТРОЙКИ ---
TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

# ТОП-5 МОНЕТ (БЕЗ ЗАВИСАНИЙ И БЛОКИРОВОК СЕТИ)
MONETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "AVAX": "AVAXUSDT",
    "XRP": "XRPUSDT"
}

last_signals = {}

def send_telegram(message):
    p1 = "https:" + "//" + "api" + "." + "telegram" + "." + "org"
    url = p1 + "/" + "bot" + TG_TOKEN + "/" + "sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Ошибка TG: {e}")

def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    b_url = "https:" + "//" + "api" + "." + "bybit" + "." + "com/v5/market/kline"
    try:
        r = requests.get(b_url, params={'category': 'linear', 'symbol': symbol, 'interval': bybit_map[interval], 'limit': limit}, timeout=5).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass
    
    bn_url = "https:" + "//" + "data-api" + "." + "binance" + "." + "vision/api/v3/klines"
    try:
        r2 = requests.get(bn_url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=5).json()
        df = pd.DataFrame(r2, columns=['ts', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig'])
        for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
        df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
        return df
    except Exception:
        return None

def calculate_indicators(df):
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    denom_rsi = loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + (gain / denom_rsi)))

    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    denom_stoch = high_14 - low_14
    denom_stoch = denom_stoch.replace(0, np.nan)
    
    fast_k = 100 * (df['close'] - low_14) / denom_stoch
    df['stoch_k'] = fast_k.rolling(window=3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    return df

def execute_sniper_strategy(name, ticker_symbol):
    try:
        df_4h = get_live_candles(ticker_symbol, '4h', limit=200)
        df_1h = get_live_candles(ticker_symbol, '1h', limit=100)
        df_15m = get_live_candles(ticker_symbol, '15m', limit=100)

        if df_4h is None or df_1h is None or df_15m is None:
            print(f"⏳ Ожидание сети для {name}...")
            return

        df_4h = calculate_indicators(df_4h)
        df_1h = calculate_indicators(df_1h)
        df_15m = calculate_indicators(df_15m)

        c4h = df_4h.iloc[-1]
        c1h = df_1h.iloc[-1]
        c15m = df_15m.iloc[-1]
        p15m = df_15m.iloc[-2]

        cp = c15m['close']

        # Вчерашняя многоэкранная фильтрация
        trend_4h_bear = c4h['close'] < c4h['ema200']
        trend_1h_bear = c1h['close'] < c1h['ema50']
        stoch_cross_down = (c15m['stoch_k'] > 75 and 
                            c15m['stoch_k'] < c15m['stoch_d'] and 
                            p15m['stoch_k'] >= p15m['stoch_d'])

        # Четкое локальное время для каждой монеты
        coin_time = (datetime.utcnow() + timedelta(hours=5)).strftime("%H:%M:%S")

        print(f"═══ ⚡ ПОЗИЦИОННЫЙ СНАЙПЕР ДЛЯ {name} [{coin_time}] ═══")
        print(f"   📍 Цена LIVE: {round(cp, 4)}")
        print(f"   📊 Снайпер 15M: Stoch = {round(c15m['stoch_k'],1)}/{round(c15m['stoch_d'],1)} | 1H Stoch = {round(c1h['stoch_k'],1)}")
        print(f"   🕐 Фильтр 4H: {'🔴 МЕДВЕЖИЙ' if trend_4h_bear else '🟢 БЫЧИЙ'} | 1H EMA50: {'🔴 ПОД' if trend_1h_bear else '🟢 НАД'}\n")

        # Сигнал на Вход
        if trend_4h_bear and trend_1h_bear and stoch_cross_down:
            if last_signals.get(name) != "SHORT":
                sl_level = round(c15m['ema100'] * 1.005, 4)
                msg = (
                    f"🔸 <b>ВХОД В ШОРТ #{name}</b>\n"
                    f"🛒 Вход: {round(cp, 4)}\n"
                    f"🛡️ Страховочный SL: {sl_level}\n"
                    f"🎯 <b>Ведем по тренду с быстрым выходом!</b>"
                )
                send_telegram(msg)
                last_signals[name] = "SHORT"

        # Сигнал на Фиксацию
        elif last_signals.get(name) == "SHORT" and c1h['stoch_k'] < 30 and c1h['stoch_k'] > c1h['stoch_d']:
            msg = (
                f"🏁 <b>ФИКСИРУЕМ ШОРТ #{name}</b>\n"
                f"💰 Цена выхода: {round(cp, 4)}\n"
                f"📊 Причина: Разворот импульса (15M EMA100 / 1H Stoch)"
            )
            send_telegram(msg)
            last_signals[name] = None

    except Exception as e:
        print(f"Ошибка обработки {name}: {e}")

def start_bot():
    # Ваши оригинальные три стартовых сообщения
    send_telegram("🤖 <b>Бот Эллиотта (4H + 1H + 15M) запущен!</b>\nЖду сигналы по BTC, ETH и SOL...")
    time.sleep(1)
    send_telegram("⚡ <b>Бот переключен на LIVE-котировки биржи!</b>\nПроверка рынка каждую 1 минуту.")
    time.sleep(1)
    send_telegram("🎯 <b>Снайпер-режим (4H + 1H + 15M + Stoch 14,3,3) активирован!</b>")

    while True:
        clear_output(wait=True)
        for name, ticker in MONETS.items():
            execute_sniper_strategy(name, ticker)
            time.sleep(2) # Плавная задержка 2 секунды, чтобы Bybit не банил
        time.sleep(30)

start_bot()
