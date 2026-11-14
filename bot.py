import os
import json
import time
import requests
import pandas as pd
import numpy as np

# Секреты читаются автоматически из настроек GitHub (Secrets)
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# --- СПИСОК ИЗ 10 МОНЕТ (МОЖЕТЕ МЕНЯТЬ ПО СВОЕМУ ВКУСУ) ---
MONETS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT',
    'AVAX': 'AVAXUSDT',
    'INJ': 'INJUSDT',
    'AAVE': 'AAVEUSDT',
    'ARB': 'ARBUSDT',
    'JUP': 'JUPUSDT',
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
        print("⚠️ Не заданы TG_TOKEN или TG_CHAT_ID в GitHub Secrets!")
        return
    url = f"https://telegram.org{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Ошибка TG: {e}")


def get_live_candles(symbol, interval, limit=300):
    bybit_map = {'4h': '240', '1h': '60', '15m': '15'}
    b_url = "https://bybit.com"
    try:
        r = requests.get(
            b_url,
            params={'category': 'linear', 'symbol': symbol, 'interval': bybit_map[interval], 'limit': limit},
            timeout=5
        ).json()
        if r.get('retCode') == 0 and r['result']['list']:
            rows = r['result']['list'][::-1]
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            for c in ['open', 'high', 'low', 'close']:
                df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['ts'].astype(int), unit='ms') + pd.Timedelta(hours=5)
            return df
    except Exception:
        pass

    bn_url = "https://binance.vision"
    r2 = requests.get(bn_url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=5).json()
    df = pd.DataFrame(r2, columns=['ts', 'open', 'high', 'low', 'close', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'ig'])
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
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
        is_high = df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max()
        is_low = df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min()
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
    return False, f"Нарушены аксиомы (W2:{r1}|W3:{r2}|W4:{r3 or diag})", diag


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
            trend_4h, t_txt = "LONG", f"🟢 4H РАСТЕТ (Выше EMA200: {round(e200_4h, 4)})"
        elif wave_4h == "SHORT" and not above_4h_ema:
            trend_4h, t_txt = "SHORT", f"🔴 4H ПАДАЕТ (Ниже EMA200: {round(e200_4h, 4)})"
        else:
            trend_4h, t_txt = "WAIT", f"⚠️ 4H ОЖИДАНИЕ (Конфликт волн и EMA200: {round(e200_4h, 4)})"

        df_1h = calculate_indicators(get_live_candles(ticker_symbol, '1h', limit=300))
        c1h, p1h = df_1h.iloc[-1], df_1h.iloc[-2]
        e200_1h = c1h['ema200']
        rsi_1h = round(c1h['rsi'], 1)
        sk_1h, sd_1h = round(c1h['stoch_k'], 1), round(c1h['stoch_d'], 1)
        psk_1h, psd_1h = p1h['stoch_k'], p1h['stoch_d']
        b_1h, s_1h = c1h['close'] > e200_1h, c1h['close'] < e200_1h

        df_15m = calculate_indicators(get_live_candles(ticker_symbol, '15m', limit=350))
        piv_15m = get_wave_pivots(df_15m, WINDOW_15M)
        curr, prev = df_15m.iloc[-1], df_15m.iloc[-2]
        cp, rsi_v = round(float(curr['close']), 4), round(curr['rsi'], 1)
        sk, sd = round(curr['stoch_k'], 1), round(curr['stoch_d'], 1)
        psk, psd = prev['stoch_k'], prev['stoch_d']

        stoch_cross_up = (sk < 25) and (sk > sd) and (psk <= psd or psk < 25)
        stoch_cross_down = (sk > 75) and (sk < sd) and (psk >= psd or psk > 75)
        stoch_txt = (
            f"🟢 Разворот вверх ({sk}/{sd})" if stoch_cross_up
            else (f"🔴 Разворот вниз ({sk}/{sd})" if stoch_cross_down else f"⚪ {sk}/{sd}")
        )

        e_bull = (cp > curr['ema50']) and (curr['ema50'] > curr['ema100'] > curr['ema200'])
        e_bear = (cp < curr['ema50']) and (curr['ema50'] < curr['ema100'] < curr['ema200'])
        e_st = "🟢 Бычий веер" if e_bull else ("🔴 Медвежий веер" if e_bear else "🟡 Смешанные EMA")
        h1_st = f"🟢 Выше EMA200 ({round(e200_1h, 4)})" if b_1h else f"🔴 Ниже EMA200 ({round(e200_1h, 4)})"

        print(f"\n═══ ⚡ ПОЗИЦИОННЫЙ СНАЙПЕР (4H + 1H + 15M) ДЛЯ {name} ═══")
        print(f"   🌍 1. Тренд 4H: {t_txt}")
        print(f"   🕐 2. Фильтр 1H: {h1_st} | Stoch 1H = {sk_1h}/{sd_1h} | RSI 1H = {rsi_1h}")
        print(f"   📊 3. Снайпер 15M: RSI = {rsi_v} | Stoch = {stoch_txt} | {e_st}")
        print(f"   📍 Цена LIVE: {cp}")

        summary_line = f"• <b>{name}</b> (${cp}): 4H {trend_4h} | 15M RSI {rsi_v}"

        # -----------------------------------------------------------------
        # 1. БЫСТРЫЙ УМНЫЙ ВЫХОД ИЗ АКТИВНОЙ СДЕЛКИ
        # -----------------------------------------------------------------
        pos = active_trades.get(name)
        if pos:
            side, entry_p, be_done, sl_p = pos['side'], pos['entry'], pos['be'], pos['sl']
            if side == "LONG":
                profit_pct = (cp - entry_p) / entry_p * 100
                if profit_pct >= 1.5 and not be_done:
                    pos['be'] = True
                    pos['sl'] = entry_p
                    send_telegram(
                        f"🛡️ <b>БЕЗУБЫТОК ПО #{name} (+{round(profit_pct, 1)}%)</b>\n"
                        f"Передвиньте Stop-Loss на цену входа: <code>{entry_p}</code>"
                    )
                if cp <= sl_p:
                    send_telegram(f"🛑 <b>STOP-LOSS ПО #{name} (LONG)</b>\nЦена: <code>{cp}</code> | Итог: {round(profit_pct, 2)}%")
                    del active_trades[name]
                    return summary_line
                if (cp < curr['ema100']) or (sk_1h > 80 and sk_1h < sd_1h and psk_1h >= psd_1h):
                    send_telegram(
                        f"💰 <b>ФИКСАЦИЯ / ВЫХОД ПО #{name} (LONG)</b>\n"
                        f"Цена: <code>{cp}</code> | Результат: <b>{round(profit_pct, 2)}%</b>\n"
                        f"Причина: разворот индикаторов (15M EMA100 / 1H Stoch)"
                    )
                    del active_trades[name]
                    return summary_line

            elif side == "SHORT":
                profit_pct = (entry_p - cp) / entry_p * 100
                if profit_pct >= 1.5 and not be_done:
                    pos['be'] = True
                    pos['sl'] = entry_p
                    send_telegram(
