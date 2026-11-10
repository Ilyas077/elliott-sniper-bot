import time
import datetime
import telebot
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. НАСТРОЙКИ ---
TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", 
    "AVAXUSDT", "LDOUSDT", "JTOUSDT", "INJUSDT", 
    "AAVEUSDT", "PUMPUSDT", "ARBUSDT", "JUPUSDT"
]

bot = telebot.TeleBot(TG_TOKEN)

# --- 2. ВСТРОЕННЫЕ ФОРМУЛЫ ИНДИКАТОРОВ ---
def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_stoch(df, k=14, d=3, smooth_k=3):
    low_min = df['low'].rolling(window=k).min()
    high_max = df['high'].rolling(window=k).max()
    fast_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_k = fast_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d

# --- 3. ПОЛУЧЕНИЕ ДАННЫХ С БИРЖИ (BYBIT + РЕЗЕРВ GATE.IO) ---
def get_market_klines(symbol, interval, limit=200):
    # Канал 1: Bybit
    url_bybit = "https://bybit.com"
    params_bybit = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url_bybit, params=params_bybit, timeout=10).json()
        if res.get('retCode') == 0 and res['result']['list']:
            list_data = res['result']['list']
            df = pd.DataFrame(list_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.iloc[::-1].reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass

    # Канал 2: Резерв Gate.io
    bar = "1h" if interval == "60" else "15m"
    pair_gate = symbol.replace("USDT", "_USDT")
    url_gate = "https://gateio.ws"
    params_gate = {"currency_pair": pair_gate, "interval": bar, "limit": limit}
    try:
        res_gate = requests.get(url_gate, params=params_gate, timeout=10).json()
        if isinstance(res_gate, list) and len(res_gate) > 0:
            df = pd.DataFrame(res_gate, columns=['timestamp', 'volume', 'close', 'high', 'low', 'open', 'extra'])
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
        
    return None

# --- 4. КОНТРОЛЬ ЛОКОМОТИВА (BTC) ---
def get_btc_global_status():
    df_btc_1h = get_market_klines("BTCUSDT", "60", limit=200)
    if df_btc_1h is None or df_btc_1h.empty:
        return "NEUTRAL", "Связь с биржей потеряна", 0
        
    df_btc_1h['EMA_200'] = calc_ema(df_btc_1h['close'], length=200)
    last_btc = df_btc_1h.iloc[-1]
    btc_price = last_btc['close']
    
    if btc_price > last_btc['EMA_200']:
        return "BULLISH", "🟢 БЫЧИЙ (Лонг разрешен)", btc_price
    else:
        return "BEARISH", "🔴 МЕДВЕЖИЙ (Только шорты)", btc_price

# --- 5. АНАЛИЗ АЛЬТКОИНОВ ---
def analyze_market(symbol, btc_status):
    df_1h = get_market_klines(symbol, "60", limit=200)   
    df_15m = get_market_klines(symbol, "15", limit=200) 
    
    if df_1h is None or df_15m is None or df_15m.empty or df_1h.empty:
        return None, None

    df_1h['EMA_200'] = calc_ema(df_1h['close'], length=200)
    trend = "UP" if df_1h.iloc[-1]['close'] > df_1h.iloc[-1]['EMA_200'] else "DOWN"

    df_15m['RSI'] = calc_rsi(df_15m['close'], length=14)
    df_15m['STOCH_K'], df_15m['STOCH_D'] = calc_stoch(df_15m, k=14, d=3, smooth_k=3)

    df_15m['loc_min'] = df_15m.iloc[argrelextrema(df_15m.low.values, np.less_equal, order=5)]['low']
    df_15m['loc_max'] = df_15m.iloc[argrelextrema(df_15m.high.values, np.greater_equal, order=5)]['high']
    waves = df_15m[(df_15m['loc_min'].notnull()) | (df_15m['loc_max'].notnull())].tail(8)
    
    current_15m = df_15m.iloc[-2]  
    prev_15m = df_15m.iloc[-3]
    
    stoch_bullish_cross = (prev_15m['STOCH_K'] < prev_15m['STOCH_D']) and (current_15m['STOCH_K'] > current_15m['STOCH_D'])
    stoch_bearish_cross = (prev_15m['STOCH_K'] > prev_15m['STOCH_D']) and (current_15m['STOCH_K'] < current_15m['STOCH_D'])

    wave_c_finished_long = False
    wave_c_finished_short = False
    
    if len(waves) >= 6:
        prices = waves['close'].values
        if trend == "UP" and prices[-1] < prices[-3]: 
            wave_c_finished_long = True
        if trend == "DOWN" and prices[-1] > prices[-3]:
            wave_c_finished_short = True

    info = f"{symbol}: RSI {round(current_15m['RSI'], 1)} | Stoch {round(current_15m['STOCH_K'], 1)}"

    if wave_c_finished_long and btc_status == "BEARISH":
        return None, info
    if wave_c_finished_short and btc_status == "BULLISH":
        return None, info

    if trend == "UP" and current_15m['RSI'] < 45 and current_15m['STOCH_K'] < 25 and stoch_bullish_cross and wave_c_finished_long:
        entry = current_15m['close']
        return {"type": "🟢 МИКРО-ЛОНГ", "price": entry, "SL": round(entry * 0.997, 5), "TP": round(entry * 1.010, 5), "rsi": round(current_15m['RSI'], 1)}, info

    if trend == "DOWN" and current_15m['RSI'] > 55 and current_15m['STOCH_K'] > 75 and stoch_bearish_cross and wave_c_finished_short:
        entry = current_15m['close']
        return {"type": "🔴 МИКРО-ШОРТ", "price": entry, "SL": round(entry * 1.003, 5), "TP": round(entry * 0.990, 5), "rsi": round(current_15m['RSI'], 1)}, info

    return None, info

# --- 6. ЗАПУСК ПРОВЕРКИ ---
btc_code, btc_text, btc_price = get_btc_global_status()
signals_found = 0
market_summary = []

for symbol in SYMBOLS:
    signal, info = analyze_market(symbol, btc_code)
    if info:
        market_summary.append(info)
    if signal:
        signals_found += 1
        msg = (
            f"🎯 **ТОЧЕЧНЫЙ ВХОД: {signal['type']}**\n\n"
            f"🪙 Монета: #{symbol}\n"
            f"💵 Вход: {signal['price']}\n"
            f"🛡️ Стоп (0.3%): {signal['SL']}\n"
            f"💰 Тейк (1.0%): {signal['TP']}\n\n"
            f"📊 RSI: {signal['rsi']}\n"
            f"🚂 BTC (${btc_price}): {btc_text}"
        )
        bot.send_message(TG_CHAT_ID, msg, parse_mode="Markdown")
    time.sleep(1)

# Отправляем отчет раз в час (в начале часа), чтобы подтвердить работу 24/7
current_minute = datetime.datetime.now().minute
if signals_found == 0 and current_minute < 15:
    pulse_msg = (
        f"📡 **Пульс рынка (GitHub 24/7)**\n"
        f"🚂 **BTC (${btc_price}):** {btc_text}\n\n"
        f"Точечных входов пока нет. Показатели:\n"
        + "\n".join(market_summary[:5])
    )
    bot.send_message(TG_CHAT_ID, pulse_msg, parse_mode="Markdown")
