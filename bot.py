import os, json, time, requests
import pandas as pd
import numpy as np

TG_TOKEN = "".join((os.environ.get("TG_TOKEN") or "").split())
TG_CHAT_ID = "".join((os.environ.get("TG_CHAT_ID") or "").split())
MONETS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'AVAX': 'AVAXUSDT', 'INJ': 'INJUSDT', 'AAVE': 'AAVEUSDT', 'ARB': 'ARBUSDT', 'JUP': 'JUPUSDT', 'LDO': 'LDOUSDT'}
STATE_FILE = "state.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TG_API = "https:" + "//api.telegram.org" + "/bot"
BN_API = "https:" + "//data-api.binance.vision" + "/api/v3/klines"
CB_API = "https:" + "//://coinbase.com" + "/products/"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"active_trades": {}, "last_signals": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, ensure_ascii=False, indent=2)

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Нет TG_TOKEN или TG_CHAT_ID!")
        return
    try:
        r = requests.post(TG_API + TG_TOKEN + "/sendMessage", json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        print(f"TG {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"Сбой TG: {e}")

def get_live_candles(symbol, interval, limit=250):
    try:
        r = requests.get(BN_API, params={'symbol': symbol, 'interval': interval, 'limit': limit}, headers=HEADERS, timeout=8)
        if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 50:
            df = pd.DataFrame([row[:6] for row in r.json()], columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            return df
    except Exception: pass
    coin = symbol.replace("USDT", "")
    gran = {'4h': 14400, '1h': 3600, '15m': 900}[interval]
    r2 = requests.get(CB_API + f"{coin}-USD/candles", params={'granularity': gran}, headers=HEADERS, timeout=8).json()
    df = pd.DataFrame(r2[:limit], columns=['ts', 'low', 'high', 'open', 'close', 'vol']).iloc[::-1].reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
    return df

def calc(df):
    for s in (50, 100, 200): df[f'ema{s}'] = df['close'].ewm(span=s, adjust=False).mean()
    d = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (d.where(d > 0, 0).rolling(14).mean() / (-d.where(d < 0, 0)).rolling(14).mean())))
    l14, h14 = df['low'].rolling(14).min(), df['high'].rolling(14).max()
    df['sk'] = (100 * (df['close'] - l14) / (h14 - l14)).rolling(3).mean()
    df['sd'] = df['sk'].rolling(3).mean()
    return df

def get_pivots(df, w):
    piv = []
    for i in range(w, len(df) - w):
        if df['high'].iloc[i] == df['high'].iloc[i-w:i+w+1].max(): piv.append({'type': 'HIGH', 'price': df['high'].iloc[i]})
        elif df['low'].iloc[i] == df['low'].iloc[i-w:i+w+1].min(): piv.append({'type': 'LOW', 'price': df['low'].iloc[i]})
    clean = []
    for p in piv:
        if not clean or clean[-1]['type'] != p['type']: clean.append(p)
    return clean

def check_elliott(w, st='BULL'):
    if len(w) < 9: return False, "Мало точек"
    p0, p1, p2, p3, p4, p5, _, _, _ = w[-9:]
    if st == 'BULL' and p0['type'] == 'LOW':
        l1, l3, l5 = p1['price'] - p0['price'], p3['price'] - p2['price'], p5['price'] - p4['price']
        r1, r2, r3 = p2['price'] > p0['price'], not (l3 < l1 and l3 < l5), p4['price'] > p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] > p2['price'])
    elif st == 'BEAR' and p0['type'] == 'HIGH':
        l1, l3, l5 = p0['price'] - p1['price'], p2['price'] - p3['price'], p4['price'] - p5['price']
        r1, r2, r3 = p2['price'] < p0['price'], not (l3 < l1 and l3 < l5), p4['price'] < p1['price']
        diag = (not r3) and (l3 < l1) and (p4['price'] < p2['price'])
    else: return False, "Смещение"
    return (True, "ВАЛИДНА") if (r1 and r2 and (r3 or diag)) else (False, "Нарушены аксиомы")

def analyze(name, sym, trades, last_sig):
    try:
        df4, df1, df15 = calc(get_live_candles(sym, '4h')), calc(get_live_candles(sym, '1h')), calc(get_live_candles(sym, '15m'))
        piv4, piv15 = get_pivots(df4, 4), get_pivots(df15, 6)
        c4, c1, p1, cur, prv = df4.iloc[-1], df1.iloc[-1], df1.iloc[-2], df15.iloc[-1], df15.iloc[-2]
        w4 = "LONG"
        if len(piv4) >= 4:
            if piv4[-1]['type'] == 'LOW' and piv4[-1]['price'] < piv4[-3]['price']: w4 = "SHORT"
            elif piv4[-1]['type'] == 'HIGH' and piv4[-1]['price'] < piv4[-2]['price']: w4 = "WAIT"
        t4 = "LONG" if (w4 == "LONG" and c4['close'] > c4['ema200']) else ("SHORT" if (w4 == "SHORT" and c4['close'] < c4['ema200']) else "WAIT")
        cp, rsi1, rsi15 = round(float(cur['close']), 4), round(c1['rsi'], 1), round(cur['rsi'], 1)
        sk, sd, psk, psd = cur['sk'], cur['sd'], prv['sk'], prv['sd']
        up = (sk < 25) and (sk > sd) and (psk <= psd or psk < 25)
        dn = (sk > 75) and (sk < sd) and (psk >= psd or psk > 75)
        print(f"=== {name} | ${cp} | 4H: {t4} | 15M RSI: {rsi15} ===")
        line = f"• <b>{name}</b> (${cp}): 4H {t4} | RSI {rsi15}"

        pos = trades.get(name)
        if pos:
            is_l = pos['side'] == "LONG"
            pnl = round(((cp - pos['entry']) if is_l else (pos['entry'] - cp)) / pos['entry'] * 100, 2)
            if pnl >= 1.5 and not pos['be']:
                pos['be'], pos['sl'] = True, pos['entry']
                send_telegram(f"🛡️ <b>БЕЗУБЫТОК #{name} (+{pnl}%)</b>\nSL: <code>{pos['entry']}</code>")
            if (is_l and cp <= pos['sl']) or (not is_l and cp >= pos['sl']):
                send_telegram(f"🛑 <b>STOP-LOSS #{name} ({pos['side']})</b>\nЦена: <code>{cp}</code> | {pnl}%")
                del trades[name]
                return line
            if (is_l and (cp < cur['ema100'] or (c1['sk'] > 80 and c1['sk'] < c1['sd']))) or (not is_l and (cp > cur['ema100'] or (c1['sk'] < 20 and c1['sk'] > c1['sd']))):
                send_telegram(f"💰 <b>ВЫХОД #{name} ({pos['side']})</b>\nЦена: <code>{cp}</code> | <b>{pnl}%</b>")
                del trades[name]
            return line

        ts = str(cur['ts'])
        if last_sig.get(name) == ts: return line
        if t4 == "LONG" and c1['close'] > c1['ema200'] and rsi1 < 68 and up and cp > cur['ema50']:
            ok, msg = check_elliott(piv15, 'BULL')
            sl = round(float(min(cur['ema200'], df15['low'].iloc[-5:].min())) * 0.998, 4)
            tp = round(cp + (cp - sl) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "LONG", "entry": cp, "sl": sl, "be": False}, ts
            send_telegram(f"🚀 <b>LONG #{name}</b>\nВход: <code>{cp}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\nRSI: {rsi1}/{rsi15} | Волны: {msg} ({'✅' if ok else '⚠️'})")
        elif t4 == "SHORT" and c1['close'] < c1['ema200'] and rsi1 > 32 and dn and cp < cur['ema50']:
            ok, msg = check_elliott(piv15, 'BEAR')
            sl = round(float(max(cur['ema200'], df15['high'].iloc[-5:].max())) * 1.002, 4)
            tp = round(cp - (sl - cp) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "SHORT", "entry": cp, "sl": sl, "be": False}, ts
            send_telegram(f"🔻 <b>SHORT #{name}</b>\nВход: <code>{cp}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\nRSI: {rsi1}/{rsi15} | Волны: {msg} ({'✅' if ok else '⚠️'})")
        return line
    except Exception as e:
        print(f"Ошибка {name}: {e}")
        return f"• <b>{name}</b>: ошибка"

if __name__ == "__main__":
    st = load_state()
    res_list = [r for n, s in MONETS.items() if (r := analyze(n, s, st["active_trades"], st["last_signals"])) and not time.sleep(1.5)]
    save_state(st)
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram("📡 <b>Тестовый запуск Снайпера (GitHub) прошел!</b>\n\n" + "\n".join(res_list))
