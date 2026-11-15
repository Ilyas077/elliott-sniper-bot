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

# Точный Stoch RSI как на вашем графике TradingView (10, 3, 14, 14, ohlc4)
def calc(df):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    df['zlsma'] = calc_zlsma_tv(ohlc4, 32)
    d = ohlc4.diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    min_r, max_r = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch_r = 100 * (rsi - min_r) / (max_r - min_r).replace(0, 1e-9)
    df['sk'] = stoch_r.rolling(10).mean()
    df['sd'] = df['sk'].rolling(3).mean()
    df['rsi'] = rsi
    return df

def analyze(name, sym, trades, last_sig):
    try:
        df1, df30 = calc(get_live_candles(sym, '1h')), calc(get_live_candles(sym, '30m'))
        c1, p1 = df1.iloc[-2], df1.iloc[-3]
        live_p = round(float(df30.iloc[-1]['close']), 4)
        cur, prv = df30.iloc[-2], df30.iloc[-3]

        h1_bull = (c1['close'] >= c1['zlsma']) and (c1['zlsma'] > p1['zlsma'])
        h1_bear = (c1['close'] <= c1['zlsma']) and (c1['zlsma'] < p1['zlsma'])
        trend_1h = "LONG" if h1_bull else ("SHORT" if h1_bear else "WAIT")

        cp, pp = round(float(cur['close']), 4), float(prv['close'])
        zl, pzl = round(float(cur['zlsma']), 4), float(prv['zlsma'])
        sk, sd = round(float(cur['sk']), 1), round(float(cur['sd']), 1)
        psk, psd = float(prv['sk']), float(prv['sd'])

        # 1. Пробой и закрепление за 30M ZLSMA 32
        cross_up_zl = (pp <= pzl) and (cp > zl) and (zl >= pzl * 0.999)
        cross_dn_zl = (pp >= pzl) and (cp < zl) and (zl <= pzl * 1.001)

        # 2. Откат внутри тренда + разворот Stoch RSI
        pullback_long = (trend_1h == "LONG") and (cp > zl) and (zl > pzl) and (sk > sd) and (psk <= psd) and (sk < 65)
        pullback_short = (trend_1h == "SHORT") and (cp < zl) and (zl < pzl) and (sk < sd) and (psk >= psd) and (sk > 35)

        zl_tag = f"🟢Над ZLSMA ({zl})" if cp >= zl else f"🔴Под ZLSMA ({zl})"
        print(f"=== {name} | ${live_p} | 1H: {trend_1h} | {zl_tag} | StochRSI: {sk}/{sd} ===")
        line = f"• <b>{name}</b> (${live_p}): 1H {trend_1h} | {zl_tag} | SRSI {sk}"

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

        if (cross_up_zl and sk > sd and sk < 88) or pullback_long:
            reason = "Пробой 30M ZLSMA вверх" if cross_up_zl else "Откат по тренду 1H + ZLSMA"
            sl = round(float(min(zl, df30['low'].iloc[-5:-1].min())) * 0.997, 4)
            tp = round(live_p + (live_p - sl) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "LONG", "entry": live_p, "sl": sl, "be": False}, ts
            send_telegram(f"🚀 <b>LONG #{name}</b> ({reason})\nВход: <code>{live_p}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\n30M ZLSMA: <code>{zl}</code> | StochRSI: {sk}/{sd} | 1H: {trend_1h}")

        elif (cross_dn_zl and sk < sd and sk > 12) or pullback_short:
            reason = "Пробой 30M ZLSMA вниз" if cross_dn_zl else "Откат по тренду 1H + ZLSMA"
            sl = round(float(max(zl, df30['high'].iloc[-5:-1].max())) * 1.003, 4)
            tp = round(live_p - (sl - live_p) * 2.5, 4)
            trades[name], last_sig[name] = {"side": "SHORT", "entry": live_p, "sl": sl, "be": False}, ts
            send_telegram(f"🔻 <b>SHORT #{name}</b> ({reason})\nВход: <code>{live_p}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\n30M ZLSMA: <code>{zl}</code> | StochRSI: {sk}/{sd} | 1H: {trend_1h}")

        return line
    except Exception as e:
        print(f"Ошибка {name}: {e}")
        return f"• <b>{name}</b>: ошибка"

if __name__ == "__main__":
    st = load_state()
    res_list = [r for n, s in MONETS.items() if (r := analyze(n, s, st["active_trades"], st["last_signals"])) and not time.sleep(1.5)]
    save_state(st)
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram("📡 <b>Снайпер подкручен под ваш StochRSI + 30M ZLSMA!</b>\n\n" + "\n".join(res_list))
