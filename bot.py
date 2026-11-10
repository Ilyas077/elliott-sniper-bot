import time
import telebot
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema

# --- НАСТРОЙКИ ---
TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

SYMBOLS = [
    "BTCUSDT",   # Локомотив (главный фильтр)
    "ETHUSDT",   # Эфириум
    "SOLUSDT",   # Солана
    "XRPUSDT",   # Рипл
    "AVAXUSDT",  # Авакс
    "LDOUSDT",   # ЛДО
    "JTOUSDT",   # JTO
    "INJUSDT",   # Injective
    "AAVEUSDT",  # Aave
    "PUMPUSDT",  # Pump
    "ARBUSDT",   # Arbitrum
    "JUPUSDT"    # Jupiter
]

bot = telebot.TeleBot(TG_TOKEN)

def get_bybit_klines(symbol, interval, limit=200):
    url = f"https://bybit.com{symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url).json()
        if response['retCode'] == 0:
            list_data = response['result']['list']
            df = pd.DataFrame(list_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.iloc[::-1].reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            return df
    except Exception as e:
        print(f"Ошибка получения данных для {symbol}: {e}")
    return None

def get_btc_global_status():
    df_btc_1h = get_bybit_klines("BTCUSDT", "60", limit=200)
    if df_btc_1h is None or df_btc_1h.empty:
        return "NEUTRAL", "Связь с биржей потеряна"
        
    df_btc_1h['EMA_200'] = ta.ema(df_btc_1h['close'], length=200)
    last_btc = df_btc_1h.iloc[-1]
    btc_price = last_btc['close']
    
    if btc_price > last_btc['EMA_200']:
        return "BULLISH", "🟢 ТРЕНД БЫЧИЙ (Альткоины можно лонговать)"
    else:
        return "BEARISH", "🔴 ТРЕНД МЕДВЕЖИЙ (Безопасны только шорты)"

def analyze_market(symbol, btc_status):
    df_1h = get_bybit_klines(symbol, "60", limit=200)   
    df_15m = get_bybit_klines(symbol, "15", limit=200) 
    
    if df_1h is None or df_15m is None or df_15m.empty or df_1h.empty:
        return None

    df_1h['EMA_200'] = ta.ema(df_1h['close'], length=200)
    trend = "UP" if df_1h.iloc[-1]['close'] > df_1h.iloc[-1]['EMA_200'] else "DOWN"

    df_15m['RSI'] = ta.rsi(df_15m['close'], length=14)
    stoch = ta.stoch(df_15m['high'], df_15m['low'], df_15m['close'], k=14, d=3, smooth_k=3)
    df_15m['STOCH_K'] = stoch['STOCHk_14_3_3']
    df_15m['STOCH_D'] = stoch['STOCHd_14_3_3']

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

    if wave_c_finished_long and btc_status == "BEARISH":
        return None
    if wave_c_finished_short and btc_status == "BULLISH":
        return None

    if trend == "UP" and current_15m['RSI'] < 42 and current_15m['STOCH_K'] < 20 and stoch_bullish_cross and wave_c_finished_long:
        entry = current_15m['close']
        sl = entry * 0.997
        tp = entry * 1.010
        return {"type": "🟢 МИКРО-ЛОНГ", "price": entry, "SL": round(sl, 5), "TP": round(tp, 5), "rsi": round(current_15m['RSI'], 1)}

    if trend == "DOWN" and current_15m['RSI'] > 58 and current_15m['STOCH_K'] > 80 and stoch_bearish_cross and wave_c_finished_short:
        entry = current_15m['close']
        sl = entry * 1.003
        tp = entry * 0.990
        return {"type": "🔴 МИКРО-ШОРТ", "price": entry, "SL": round(sl, 5), "TP": round(tp, 5), "rsi": round(current_15m['RSI'], 1)}

    return None

# --- ЗАПУСК ОДИНОЧНОЙ ПРОВЕРКИ (ДЛЯ GITHUB ACTIONS) ---
print("Запуск сканирования рынка...")
btc_code, btc_text = get_btc_global_status()
signals_found = 0

for symbol in SYMBOLS:
    signal = analyze_market(symbol, btc_code)
    if signal:
        signals_found += 1
        msg = (
            f"🎯 **ТОЧЕЧНЫЙ СКАЛЬП-ВХОД: {signal['type']}**\n\n"
            f"🪙 Монета: #{symbol}\n"
            f"💵 Цена входа: {signal['price']}\n"
            f"🛡️ Стоп-Лосс (0.3%): {signal['SL']}\n"
            f"💰 Тейк-Профит (1.0%): {signal['TP']}\n\n"
            f"📊 RSI альты: {signal['rsi']}\n"
            f"🚂 Локомотив BTC: {btc_text}\n\n"
            f"💡 *Защита депозита:* При +0.5% перенеси стоп в безубыток!"
        )
        try:
            bot.send_message(TG_CHAT_ID, msg, parse_mode="Markdown")
        except Exception as e:
            print(f"Ошибка отправки для {symbol}: {e}")
    time.sleep(1)

print(f"Проверка завершена. Найдено сигналов: {signals_found}")
