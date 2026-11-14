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
    if not TG_TOKEN or not TG_CHAT_ID: return
    try:
        r = requests.post(TG_API + TG_TOKEN + "/sendMessage", json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        print(f"TG {r.status_code}: {r.text[:80]}")
    except Exception as e: print(f"Сбой TG: {e}")

def get_live_candles(symbol, interval, limit=150):
    try:
        r = requests.get(BN_API, params={'symbol': symbol, 'interval': interval, 'limit': limit}, headers=HEADERS, timeout=8)
        if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 50:
            df = pd.DataFrame([row[:6] for row in r.json()], columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
            return df
    except Exception: pass
    coin = symbol.replace("USDT", "")
    gran = 3600 if interval == '1h' else 900
    req_lim = limit if interval == '1h' else limit * 2
    r2 = requests.get(CB_API + f"{coin}-USD/candles", params={'granularity': gran}, headers=HEADERS, timeout=8).json()
    df = pd.DataFrame(r2[:req_lim], columns=['ts', 'low', 'high', 'open', 'close', 'vol']).iloc[::-1].reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
    if interval == '1h': return df
    df['dt'] = pd.to_datetime(df['ts'].astype(int), unit='s')
    df30 = df.set_index('dt').resample('30min').agg({'ts': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'}).dropna().reset_index(drop=True)
    return df30.tail(limit).reset_index(drop=True)

def calc_zlsma_tv(series, length=32):
    w = np.arange(1, length + 1)
    def lsma(s):
        wma = s.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
        sma = s.rolling(length).mean()
        return 3 * wma - 2 * sma
    l1 = lsma(series)
    l2 = lsma(l1)
    return l1 + (l1 - l2)

def calc(df):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    df['zlsma'] = calc_zlsma_tv(ohlc4, 32)
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
        df1, df30 = calc(get_live_candles(sym, '1h')), calc(get_live_candles(sym, '30m'))
        piv30 = get_pivots(df30, 5)
        c1, p1 = df1.iloc[-2], df1.iloc[-3]
        live_p = round(float(df30.iloc[-1]['close']), 4)
        cur, prv = df30.iloc[-2], df30.iloc[-3]

        # Ориентир 1H по ZLSMA 32
        h1_bull = (c1['close'] >= c1['zlsma']) and (c1['zlsma'] > p1['zlsma'])
        h1_bear = (c1['close'] <= c1['zlsma']) and (c1['zlsma'] < p1['zlsma'])
        trend_1h = "LONG" if h1_bull else ("SHORT" if h1_bear else "WAIT")

        # Вход и выход 30M по ZLSMA 32 + разворот Стохастика
        cp, rsi1, rsi30 = round(float(cur['close']), 4), round(c1['rsi'], 1), round(cur['rsi'], 1)
        zl, pzl = round(float(cur['zlsma']), 4), float(prv['zlsma'])
        in_long_zl = (cp >= zl) and (cur['high'] >= zl) and (zl > pzl)
        in_short_zl = (cp <= zl) and (cur['low'] <= zl) and (zl < pzl)

        sk, sd, psk, psd = cur['sk'], cur['sd'], prv['sk'], prv['sd']
        up = (sk < 40) and (sk > sd) and (psk <= psd or psk < 40)
        dn = (sk > 60) and (sk < sd) and (psk >= psd or psk > 60)

        zl_tag = f"🟢Над 30M ZLSMA ({zl})" if cp >= zl else f"🔴Под 30M ZLSMA ({zl})"
        print(f"=== {name} | ${live_p} | 1H: {trend_1h} | {zl_tag} ===")
        line = f"• <b>{name}</b> (${live_p}): 1H {trend_1h} | {zl_tag}"

        pos = trades.get(name)
        if pos:
            is_l = pos['side'] == "LONG"
            pnl = round(((live_p - pos['entry']) if is_l else (pos['entry'] - live_p)) / pos['entry'] * 100, 2)
            if pnl >= 1.5 and not pos['be']:
                pos['be'], pos['sl'] = True, pos['entry']
                send_telegram(f"🛡️ <b>БЕЗУБЫТОК #{name} (+{pnl}%)</b>\nSL: <code>{pos['entry']}</code>")
            if (is_l and live_p <= pos['sl']) or (not is_l and live_p >= pos['sl']):
                send_telegram(f"🛑 <b>STOP-LOSS #{name} ({pos['side']})</b>\nЦена: <code>{live_p}</code> | {pnl}%")
                del trades[name]
                return line
            # Выход при закрытии 30M свечи с обратной стороны ZLSMA 32
            if (is_l and cp < zl) or (not is_l and cp > zl):
                send_telegram(f"💰 <b>ВЫХОД ПО 30M ZLSMA #{name} ({pos['side']})</b>\nЦена: <code>{live_p}</code> (линия {zl}) | Итог: <b>{pnl}%</b>")
                del trades[name]
            return line

        ts = str(cur['ts'])
        if last_sig.get(name) == ts: return line
        if trend_1h == "LONG" and rsi1 < 72 and in_long_zl and up:
            ok, msg = check_elliott(piv30, 'BULL')
            sl = round(float(min(zl, df30['low'].iloc[-6:-1].min())) * 0.998, 4)
            tp = round(live_p + (live_p - sl) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "LONG", "entry": live_p, "sl": sl, "be": False}, ts
            send_telegram(f"🚀 <b>LONG #{name} (1H + 30M ZLSMA)</b>\nВход: <code>{live_p}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\n30M ZLSMA: <code>{zl}</code> | RSI: {rsi1}/{rsi30} | Волны: {msg} ({'✅' if ok else '⚠️'})")
        elif trend_1h == "SHORT" and rsi1 > 28 and in_short_zl and dn:
            ok, msg = check_elliott(piv30, 'BEAR')
            sl = round(float(max(zl, df30['high'].iloc[-6:-1].max())) * 1.002, 4)
            tp = round(live_p - (sl - live_p) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "SHORT", "entry": live_p, "sl": sl, "be": False}, ts
            send_telegram(f"🔻 <b>SHORT #{name} (1H + 30M ZLSMA)</b>\nВход: <code>{live_p}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\n30M ZLSMA: <code>{zl}</code> | RSI: {rsi1}/{rsi30} | Волны: {msg} ({'✅' if ok else '⚠️'})")
        return line
    except Exception as e:
        print(f"Ошибка {name}: {e}")
        return f"• <b>{name}</b>: ошибка"

if __name__ == "__main__":
    st = load_state()
    res_list = [r for n, s in MONETS.items() if (r := analyze(n, s, st["active_trades"], st["last_signals"])) and not time.sleep(1.5)]
    save_state(st)
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram("📡 <b>Снайпер переведен на 1H + 30M ZLSMA 32!</b>\n\n" + "\n".join(res_list))
