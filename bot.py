import warnings
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- НАСТРОЙКИ ПЕРВОГО БОТА (ИЗ ТВОЕГО GOOGLE COLAB) ---
TG_TOKEN = "9105916544:AAGyQAXffLgHNGcKMjJMHH43qCvvjFLHIPw"
TG_CHAT_ID = "531386055"

# ТОП-5 МОНЕТ ДЛЯ СКАНИРОВАНИЯ
MONETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "AVAX": "AVAXUSDT",
    "XRP": "XRPUSDT"
}

last_signals = {}

def send_telegram(message):
    # ИСПРАВЛЕНО: Установлен точный рабочий адрес Telegram API
    url = f"https://telegram.org{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Ошибка TG: {e}")

def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    b_url = "https://bybit.com"
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
    
    bn_url = "https://binance.vision"
    try:
        r2 = requests.get(bn_url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=5).json()
        df = pd.DataFrame(r2, columns=['ts', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig'])
        for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
        df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
        return df
    except Exception:
        return None

def calculate_indicators(df):
    if df is None or df.empty: return None
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
    
    df['stoch_k'] = (100 * (df['close'] - low_14) / denom_stoch).rolling(window=3).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    return df

def execute_sniper_strategy(name, ticker_symbol):
    try:
        df_4h = calculate_indicators(get_live_candles(ticker_symbol, '4h', limit=200))
        df_1h = calculate_indicators(get_live_candles(ticker_symbol, '1h', limit=100))
        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', limit=100))

        if df_4h is None or df_1h is None or df_15m is None:
            return

        c4h = df_4h.iloc[-1]
        c1h = df_1h.iloc[-1]
        c15m = df_15m.iloc[-1]
        p15m = df_15m.iloc[-2]

        cp = c15m['close']

        trend_4h_bear = c4h['close'] < c4h['ema200']
        trend_1h_bear = c1h['close'] < c1h['ema50']
        stoch_cross_down = (c15m['stoch_k'] > 75 and 
                            c15m['stoch_k'] < c15m['stoch_d'] and 
                            p15m['stoch_k'] >= p15m['stoch_d'])

        tz_offset = timezone(timedelta(hours=5))
        coin_time = datetime.now(tz_offset).strftime("%H:%M:%S")

        print(f"═══ ⚡ СНАЙПЕР ДЛЯ {name} [{coin_time}] LIVE: {round(cp, 4)} ═══")

        # Проверка условий на ШОРТ сигнал
        if trend_4h_bear and trend_1h_bear and stoch_cross_down:
            if last_signals.get(name) != "SHORT":
                sl_level = round(c15m['ema100'] * 1.005, 4)
                msg = (
                    f"🔻 <b>ВХОД В ШОРТ #{name} (БОТ ЭЛЛИОТТА)</b>\n\n"
                    f"🛒 Вход: <code>{round(cp, 4)}</code>\n"
                    f"🛡️ Страховочный SL: <code>{sl_level}</code>\n"
                    f"🎯 <b>Стратегия: 4H Тренд + 1H Резонанс + Отскок 15М.</b>"
                )
                send_telegram(msg)
                last_signals[name] = "SHORT"

        # Условие для фиксации и выхода из позиции
        elif last_signals.get(name) == "SHORT" and c1h['stoch_k'] < 30 and c1h['stoch_k'] > c1h['stoch_d']:
            msg = (
                f"🏁 <b>ФИКСИРУЕМ ШОРТ #{name}</b>\n\n"
                f"💰 Цена выхода: <code>{round(cp, 4)}</code>\n"
                f"📊 Причина: Разворот импульса по Стохастику."
            )
            send_telegram(msg)
            last_signals[name] = None

    except Exception as e:
        print(f"Ошибка обработки {name}: {e}")

if __name__ == "__main__":
    print("Запуск одиночного сканирования ТОП-5 монет Эллиотта...")
    
    # ИСПРАВЛЕНО: Убран бесконечный цикл while True под требования GitHub Actions
    for name, ticker in MONETS.items():
        execute_sniper_strategy(name, ticker)
        time.sleep(2)
        
    print("Сканирование успешно завершено.")
