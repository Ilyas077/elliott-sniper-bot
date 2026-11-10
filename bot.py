# 1. УСТАНОВКА НЕОБХОДИМЫХ БИБЛИОТЕК (для Google Colab)
!pip install pytelegrambotapi pandas pandas_ta scipy requests --quiet

import time
import telebot
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema

# --- 2. НАСТРОЙКИ (ПОЛНЫЙ ОБЪЕДИНЕННЫЙ СПИСОК МОНЕТ) ---
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
last_signals = {symbol: 0 for symbol in SYMBOLS}

# --- 3. ПОЛУЧЕНИЕ ДАННЫХ С БИРЖИ BYBIT ---
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

# --- 4. ГЛОБАЛЬНЫЙ КОНТРОЛЬ ЛОКОМОТИВА (BTC) ---
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

# --- 5. ТОЧЕЧНЫЙ СКАЛЬПИНГ АЛЬТКОИНОВ ---
def analyze_market(symbol, btc_status):
    df_1h = get_bybit_klines(symbol, "60", limit=200)   
    df_15m = get_bybit_klines(symbol, "15", limit=200) 
    
    if df_1h is None or df_15m is None or df_15m.empty or df_1h.empty:
        return None

    # Направление локального тренда по EMA 200 на 1 часе
    df_1h['EMA_200'] = ta.ema(df_1h['close'], length=200)
    trend = "UP" if df_1h.iloc[-1]['close'] > df_1h.iloc[-1]['EMA_200'] else "DOWN"

    # Индикаторы на младшем таймфрейме 15м
    df_15m['RSI'] = ta.rsi(df_15m['close'], length=14)
    stoch = ta.stoch(df_15m['high'], df_15m['low'], df_15m['close'], k=14, d=3, smooth_k=3)
    df_15m['STOCH_K'] = stoch['STOCHk_14_3_3']
    df_15m['STOCH_D'] = stoch['STOCHd_14_3_3']

    # Поиск экстремумов волн Эллиотта (Глубина 5 свечей)
    df_15m['loc_min'] = df_15m.iloc[argrelextrema(df_15m.low.values, np.less_equal, order=5)]['low']
    df_15m['loc_max'] = df_15m.iloc[argrelextrema(df_15m.high.values, np.greater_equal, order=5)]['high']
    waves = df_15m[(df_15m['loc_min'].notnull()) | (df_15m['loc_max'].notnull())].tail(8)
    
    current_15m = df_15m.iloc[-2]  
    prev_15m = df_15m.iloc[-3]
    
    # Пересечение Стохастика
    stoch_bullish_cross = (prev_15m['STOCH_K'] < prev_15m['STOCH_D']) and (current_15m['STOCH_K'] > current_15m['STOCH_D'])
    stoch_bearish_cross = (prev_15m['STOCH_K'] > prev_15m['STOCH_D']) and (current_15m['STOCH_K'] < current_15m['STOCH_D'])

    wave_c_finished_long = False
    wave_c_finished_short = False
    
    if len(waves) >= 6:
        prices = waves['close'].values
        # Окончание коррекционной волны С
        if trend == "UP" and prices[-1] < prices[-3]: 
            wave_c_finished_long = True
        if trend == "DOWN" and prices[-1] > prices[-3]:
            wave_c_finished_short = True

    # --- ЖЕСТКИЕ СИСТЕМНЫЕ ФИЛЬТРЫ ДЕПОЗИТА ---
    if wave_c_finished_long and btc_status == "BEARISH":
        return None  # Локомотив падает — лонг альты запрещен
    if wave_c_finished_short and btc_status == "BULLISH":
        return None  # Локомотив растет — шорт альты запрещен

    # --- ТРИГГЕРЫ ВХОДА ---
    # ЛОНГ СДЕЛКА
    if trend == "UP" and current_15m['RSI'] < 42 and current_15m['STOCH_K'] < 20 and stoch_bullish_cross and wave_c_finished_long:
        entry = current_15m['close']
        sl = entry * 0.997   # Жесткий стоп 0.3%
        tp = entry * 1.010   # Быстрый тейк 1.0%
        return {"type": "🟢 МИКРО-ЛОНГ", "price": entry, "SL": round(sl, 5), "TP": round(tp, 5), "rsi": round(current_15m['RSI'], 1)}

    # ШОРТ СДЕЛКА
    if trend == "DOWN" and current_15m['RSI'] > 58 and current_15m['STOCH_K'] > 80 and stoch_bearish_cross and wave_c_finished_short:
        entry = current_15m['close']
        sl = entry * 1.003   # Жесткий стоп 0.3%
        tp = entry * 0.990   # Быстрый тейк 1.0%
        return {"type": "🔴 МИКРО-ШОРТ", "price": entry, "SL": round(sl, 5), "TP": round(tp, 5), "rsi": round(current_15m['RSI'], 1)}

    return None

# --- 6. ГЛАВНЫЙ ЦИКЛ ОПРОСА РЫНКА ---
print("Бот-спасатель успешно запущен в Colab...")
try:
    bot.send_message(TG_CHAT_ID, f"🛡️ Бот-спасатель запущен. Отслеживаю {len(SYMBOLS)} монет с жестким фильтром по BTC.")
except Exception as e:
    print(f"Ошибка отправки стартового сообщения в Telegram: {e}")

while True:
    btc_code, btc_text = get_btc_global_status()
    
    for symbol in SYMBOLS:
        signal = analyze_market(symbol, btc_code)
        current_time = time.time()
        
        # Если сигнал есть и по этой монете не слали уведомлений последние 15 минут
        if signal and (current_time - last_signals[symbol] > 900):
            msg = (
                f"🎯 **ТОЧЕЧНЫЙ СКАЛЬП-ВХОД: {signal['type']}**\n\n"
                f"🪙 Монета: #{symbol}\n"
                f"💵 Цена входа: {signal['price']}\n"
                f"🛡️ Стоп-Лосс (0.3%): {signal['SL']}\n"
                f"💰 Тейк-Профит (1.0%): {signal['TP']}\n\n"
                f"📊 RSI альты: {signal['rsi']}\n"
                f"🚂 Локомотив BTC: {btc_text}\n\n"
                f"💡 *Защита депозита:* Как только цена даст +0.5%, руками перенеси стоп-лосс в безубыток!"
            )
            try:
                bot.send_message(TG_CHAT_ID, msg, parse_mode="Markdown")
                last_signals[symbol] = current_time 
            except Exception as e:
                print(f"Ошибка отправки сигнала для {symbol}: {e}")
            
        time.sleep(2) # Пауза между запросами монет
        
    time.sleep(20) # Пауза перед новым кругом проверки
