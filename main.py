import os
import json
import time
from datetime import datetime, time as dtime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import schedule
import pytz
import pandas_market_calendars as mcal
from dotenv import load_dotenv

load_dotenv()
# -----------------------------
# CONFIG
# -----------------------------
# Telegram (ortam değişkeni veya doğrudan buraya koyabilirsin)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", )
CHAT_ID = os.getenv("CHAT_ID", )

# Periyodik kontrol aralığı (dakika)
CHECK_INTERVAL_MINUTES = int(os.environ.get("CHECK_INTERVAL_MINUTES", 1))

# Borsa saatleri opsiyonu: True => sadece MARKET_OPEN..MARKET_CLOSE arasında kontrol yap
USE_MARKET_HOURS = os.environ.get("USE_MARKET_HOURS", "True").lower() in ("1", "true", "yes")
#USE_MARKET_HOURS = False
# Market timezone ve saatler (BIST örneği — istersen değiştir)
MARKET_TZ = os.environ.get("MARKET_TZ", "Europe/Istanbul")
MARKET_OPEN_HH = int(os.environ.get("MARKET_OPEN_HH", 10))
MARKET_OPEN_MM = int(os.environ.get("MARKET_OPEN_MM", 00))
MARKET_CLOSE_HH = int(os.environ.get("MARKET_CLOSE_HH", 18))
MARKET_CLOSE_MM = int(os.environ.get("MARKET_CLOSE_MM", 00))
MARKET_OPEN = dtime(hour=MARKET_OPEN_HH, minute=MARKET_OPEN_MM)
MARKET_CLOSE = dtime(hour=MARKET_CLOSE_HH, minute=MARKET_CLOSE_MM)

# BIST (Borsa İstanbul) takvimi
bist = mcal.get_calendar("XIST")
load_dotenv()
def is_bist_open_now(now):
    """BIST için resmi tatil + hafta sonu + saat kontrolü"""
    # Bugün için takvim
    schedule = bist.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False  # tatil veya hafta sonu

    # Senin config'te verdiğin saat aralığına bak
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

# Kaç top listesi isteriz? (her model için top N)
TOP_N = int(os.environ.get("TOP_N", 5))

# Dosya isimleri (columns.json ve data.json senin verilerin)
COLUMNS_JSON = os.environ.get("COLUMNS_JSON", "columns.json")
DATA_JSON = os.environ.get("DATA_JSON", "data.json")

# Kullanıcı bilgilendirmesi
print("CONFIG:")
print(f"  CHECK_INTERVAL_MINUTES = {CHECK_INTERVAL_MINUTES} minutes")
print(f"  USE_MARKET_HOURS = {USE_MARKET_HOURS}")
print(f"  MARKET HOURS = {MARKET_OPEN} -> {MARKET_CLOSE} ({MARKET_TZ})")
print(f"  Columns file = {COLUMNS_JSON}, Data file = {DATA_JSON}")
print("--------------------------------------------------\n")


# -----------------------------
# UTIL: Telegram & price fetch
# -----------------------------
def send_telegram_message(text: str):
    """Kısa ve güvenli Telegram gönderimi."""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("YOUR_"):
        print("[WARN] TELEGRAM_TOKEN ayarlı değil. Telegram mesajı gönderilmeyecek. Mesaj içeriği:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("[Telegram error]", r.status_code, r.text)
    except Exception as e:
        print("[Telegram exception]", e)

def safe_get_last_price(symbol: str):
    """Yahoo Finance üzerinden son kapanış fiyatını çek.
    Varsayılan BIST için '.IS' eklenir. Eğer sembol NASDAQ gibi ise, sembolü doğrudan kullan."""
    if not symbol:
        return None
    # Basit heuristic: eğer sembol içinde '.' veya '-' ya da büyük harfle NASDAQ/NYSE ise ayrı kullanım gerekebilir.
    # Burada senin verilerin BIST ise ".IS" ekliyoruz. İstersen sembol formatına göre değiştir.
    ticker_symbol = f"{symbol}.IS"
    try:
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1d", interval="1d")
        if hist is None or hist.empty:
            return None
        last_close = hist["Close"].iloc[-1]
        if pd.isna(last_close):
            return None
        return float(last_close)
    except Exception as e:
        print(f"[price fetch error] {symbol}: {e}")
        return None

def get_yahoo_price_history(symbol, period="1mo", interval="1d"):
    """Belirtilen hisse için geçmiş fiyat verilerini döndürür (Yahoo Finance)."""
    try:
        ticker_symbol = f"{symbol}.IS"
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period=period, interval=interval)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"[history fetch error] {symbol}: {e}")
        return None

# -----------------------------
# ANALYZE ONCE (ilk hesaplama)
# -----------------------------
def analyze_once():
    """columns.json ve data.json okuyup combined_df oluşturur,
       Balanced/RSI skorlarını hesaplar,
       current_price ve target price hesaplayıp top listeleri döner."""
    # --- read files
    if not os.path.exists(COLUMNS_JSON):
        raise FileNotFoundError(f"{COLUMNS_JSON} bulunamadı.")
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"{DATA_JSON} bulunamadı.")

    with open(COLUMNS_JSON, "r", encoding="utf-8") as f:
        cols_obj = json.load(f)
        columns = cols_obj.get("columns", [])

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data_obj = json.load(f)
        rows = data_obj.get("data", [])

    # --- map rows to dicts using columns
    rows_mapped = []
    for item in rows:
        sym = item.get("s", "")
        arr = item.get("d", [])
        # flatten nested lists if any
        flat = []
        for v in arr:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        # pad/truncate to length of columns
        flat = flat[: len(columns)] + [None] * max(0, len(columns) - len(flat))
        row_dict = dict(zip(columns, flat))
        row_dict["symbol"] = sym.split(":")[-1] if sym else ""
        rows_mapped.append(row_dict)

    df = pd.DataFrame(rows_mapped)

    # --- rating map (string to numeric) ---
    rating_map = {"StrongBuy": 2.0, "Buy": 1.0, "Neutral": 0.0, "Sell": -1.0, "StrongSell": -2.0}
    for col in ["TechRating_1D", "MARating_1D", "OsRating_1D"]:
        if col in df.columns:
            df[col] = df[col].map(rating_map)

    # numeric conversions
    numeric_cols = ["RSI", "Mom", "Stoch.K", "Stoch.D", "AO", "CCI20"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- RSI special score (kısmi) ---
    rsi = df["RSI"]
    # avoid warnings for all-NaN
    rsi_score = -((rsi - 60.0).abs() / 60.0)
    rsi_bonus = np.select(
        [rsi.between(50, 70, inclusive="both"), rsi.between(70, 80, inclusive="left"), rsi > 80],
        [0.6, 0.2, -0.3],
        default=-0.1,
    )
    df["rsi_score"] = rsi_score + rsi_bonus


    # Balanced and RSI weighted
    df["BalancedScore"] = df["RSI"].fillna(0) * 0.25 + df["OsRating_1D"].fillna(0) * 0.25 + df["TechRating_1D"].fillna(0) * 0.25 + df["MARating_1D"].fillna(0) * 0.25
    df["RSIWeightedScore"] = df["RSI"].fillna(0) * 0.6 + df["OsRating_1D"].fillna(0) * 0.1333 + df["TechRating_1D"].fillna(0) * 0.1333 + df["MARating_1D"].fillna(0) * 0.1333

    # --- current price for each symbol (ilk an) ---
    unique_syms = df["symbol"].unique().tolist()
    prices_map = {}
    for s in unique_syms:
        prices_map[s] = safe_get_last_price(s)

    df["current_price"] = df["symbol"].map(prices_map)

    # --- target price calculation (model-specific) ---
    def target_by_score(price, score):
        if price is None or pd.isna(price) or score is None or pd.isna(score):
            return None
        if score >= 4:
            return price * 1.15
        elif score >= 3:
            return price * 1.11
        elif score >= 2:
            return price * 1.08
        elif score >= 1:
            return price * 1.04
        elif score < 0:
            return price * 0.95
        else:
            return price

    df["target_price_balanced"] = df.apply(lambda r: target_by_score(r["current_price"], r["BalancedScore"]), axis=1)

    # RSI-target logic
    def target_by_rsi(price, rsi_val):
        if price is None or pd.isna(price) or rsi_val is None or pd.isna(rsi_val):
            return None
        if 50 <= rsi_val < 70:
            return price * 1.05
        elif 70 <= rsi_val < 80:
            return price * 1.08
        elif rsi_val >= 80:
            return price * 0.95
        elif 40 <= rsi_val < 50:
            return price * 1.03
        else:
            return price * 0.90

    df["target_price_rsi"] = df.apply(lambda r: target_by_rsi(r["current_price"], r["RSI"]), axis=1)

    # expected change formatting
    def format_expected_change(row, tp_col):
        try:
            price = row["current_price"]
            tp = row.get(tp_col, None)
            if price is None or tp is None or pd.isna(price) or pd.isna(tp):
                return "-"
            pct = (tp - price) / price * 100.0
            return f"{pct:.1f}%"
        except Exception:
            return "-"

    df["expected_change_balanced"] = df.apply(lambda r: format_expected_change(r, "target_price_balanced"), axis=1)
    df["expected_change_rsi"] = df.apply(lambda r: format_expected_change(r, "target_price_rsi"), axis=1)

    # --- pick top lists ---
    top_balanced = df.sort_values("BalancedScore", ascending=False).head(TOP_N)
    top_rsi = df.sort_values("RSIWeightedScore", ascending=False).head(TOP_N)

    # terminal output (detay)
    print("\n=== İlk Analiz - Top Lists (terminal output) ===")
    def print_top(df_, model_name):
        cols_to_show = ["symbol", f"current_price", f"target_price_{model_name.lower()}", f"expected_change_{model_name.lower()}"]
        print(f"\n--- {model_name} Top {TOP_N} ---")
        # bazı sütunlar eksikse esnek davran
        show_cols = [c for c in cols_to_show if c in df_.columns]
        if df_.empty:
            print("boş")
            return
        print(df_[show_cols].to_string(index=False))
    print_top(top_balanced, "balanced")
    print_top(top_rsi, "rsi")

    # Telegram initial message (tek mesajda üç liste)
    def make_initial_message(top_bal, top_rsi):
        msg = "📌 İlk analiz sonuçları — Takip edilecek hisseler (top lists):\n\n"
        for df_top, model in [ (top_bal, "Balanced"), (top_rsi, "RSI")]:
            msg += f"📊 {model} Top {TOP_N}:\n"
            if df_top.empty:
                msg += " (yok)\n\n"
                continue
            for i, r in df_top.iterrows():
                sym = r["symbol"]
                bal = f"{r['BalancedScore']:.2f}" if pd.notna(r.get("BalancedScore")) else "-"
                price = f"{r['current_price']:.2f}" if pd.notna(r.get("current_price")) else "-"
                # target col name consistent
                tp_field = f"target_price_{model.lower()}"
                tp = f"{r.get(tp_field):.2f}" if pd.notna(r.get(tp_field)) else "-"
                exp_field = f"expected_change_{model.lower()}"
                exp = r.get(exp_field, "-")
                msg += f"{sym} | Price:{price} | Target:{tp} | Δ:{exp}\n"
            msg += "\n"
        return msg

    initial_msg = make_initial_message(top_balanced, top_rsi)
    send_telegram_message(initial_msg)

    # prepare monitored dict (initial baseline + targets + flags)
    monitored = {}
    for df_top in ( top_balanced, top_rsi):
        for _, row in df_top.iterrows():
            sym = row["symbol"]
            if not sym or pd.isna(sym):
                continue
            if sym not in monitored:
                monitored[sym] = {
                    "baseline_price": float(row["current_price"]) if pd.notna(row.get("current_price")) else None,
                    "target_price_balanced": float(row["target_price_balanced"]) if pd.notna(row.get("target_price_balanced")) else None,
                    "target_price_rsi": float(row["target_price_rsi"]) if pd.notna(row.get("target_price_rsi")) else None,
                    "alerts": {"balanced": False, "rsi": False},
                    "last_movement_dir": None,  # "up"/"down"/None
                }
            else:
                # update missing targets if any
                if pd.notna(row.get("target_price_balanced")):
                    monitored[sym]["target_price_balanced"] = float(row["target_price_balanced"])
                if pd.notna(row.get("target_price_rsi")):
                    monitored[sym]["target_price_rsi"] = float(row["target_price_rsi"])

    # return full df + monitored dictionary + top lists
    top_dict = { "Balanced": top_balanced, "RSI": top_rsi}
    return df, monitored, top_dict


# -----------------------------
# Price checker factory
# -----------------------------

def create_price_checker(monitored_dict):
    # --- Trend verilerini saklayacak önbellek ---
    trend_cache = {}  # örnek: { "ASELS": {"timestamp": datetime, "rsi": 63, "ema10": ..., "ema20": ..., "macd": ..., "obv": ..., "obv_slope": ...} }

    def get_trend_data(symbol):
        """Trend göstergelerini getirir, 10 dakikada bir yeniler."""
        now = datetime.now()
        cache = trend_cache.get(symbol)
        if cache and (now - cache["timestamp"]).seconds < 600:  # 10 dakika
            return cache

        try:
            ticker_symbol = f"{symbol}.IS"
            t = yf.Ticker(ticker_symbol)
            # günlük 1 aylık veriyi alıyoruz (günlük baz)
            data = t.history(period="1mo", interval="1d")
            if data is None or data.empty:
                return None

            # EMA, RSI, MACD, OBV hesapla
            data["EMA10"] = data["Close"].ewm(span=10, adjust=False).mean()
            data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
            delta = data["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))
            data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
            # OBV
            data["OBV"] = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

            latest = data.iloc[-1]

            # obv slope: son 3 günlük OBV değişiminin ortalaması -> hızlı hacim yönü göstergesi
            obv_slope = None
            try:
                obv_recent = data["OBV"].dropna().values
                if len(obv_recent) >= 3:
                    obv_slope = float(np.mean(np.diff(obv_recent[-3:])))
                else:
                    obv_slope = float(np.diff(obv_recent).mean()) if len(obv_recent) >= 2 else 0.0
            except Exception:
                obv_slope = 0.0

            trend_info = {
                "timestamp": now,
                "rsi": float(latest.get("RSI", np.nan)),
                "ema10": float(latest.get("EMA10", np.nan)),
                "ema20": float(latest.get("EMA20", np.nan)),
                "macd": float(latest.get("MACD", np.nan)),
                "obv": float(latest.get("OBV", 0.0)),
                "obv_slope": obv_slope,
                # gerekirse ilave seri veya değerler eklenebilir
            }
            trend_cache[symbol] = trend_info
            return trend_info
        except Exception as e:
            print(f"⚠️ {symbol} trend datası alınamadı:", e)
            return None

    def analyze_trend_with_strength(rsi, ema10, ema20, obv_slope, recent_prices=None):
        """
        Daha dengeli trend belirleme ve 'trend güç' puanı döner.
        Döndürülen: (trend_label, advice_pair, trend_strength_int)
        """
        # defaultlar
        trend_label = "⏸ Kararsız"
        advice_pair = {
            "own": "Veri yetersiz. Hacim ve fiyatı izlemeye devam et.",
            "no_own": "Veri yetersiz. Giriş için teyit bekle."
        }
        # basit göstergeler
        ema_diff = ema10 - ema20
        # momentum consistency (opsiyonel): son 4 kapanış yönü
        consistency = 0
        if recent_prices is not None and len(recent_prices) >= 4:
            seq = np.sign(np.diff(recent_prices[-4:])).tolist()
            consistency = sum(seq)  # +3..-3

        # trend strength skoru (0-10)
        strength = 5
        if ema_diff > 0:
            strength += 2
        if ema_diff > (0.01 * (recent_prices[-1] if recent_prices else 1)):  # anlamlı pozitif fark
            strength += 1
        if rsi is not None and rsi > 60:
            strength += 1
        if obv_slope is not None and obv_slope > 0:
            strength += 1
        if consistency >= 2:
            strength += 1
        strength = max(0, min(10, int(strength)))

        # karar (daha yumuşak eşikler)
        if ( (consistency >= 2 and ema_diff > 0) or (ema_diff > 0 and obv_slope > 0 and rsi and rsi > 52) or (strength >= 7 and ema_diff > 0) ):
            trend_label = "📈 Yükseliş (güç: {}/10)".format(strength)
            advice_pair = {
                "own": "Trend olumlu. Pozisyonu koru; kademeli alım için geri çekilmeleri %2-%4 aralığında düşünebilirsin.",
                "no_own": "Momentum pozitif. Hacim teyit ediyorsa küçük miktarda giriş düşünülebilir."
            }
        elif ( (consistency <= -2 and ema_diff < 0) or (ema_diff < 0 and obv_slope < 0 and rsi and rsi < 48) or (strength <= 3 and ema_diff < 0) ):
            trend_label = "📉 Düşüş (güç: {}/10)".format(strength)
            advice_pair = {
                "own": "Trend aşağı yönlü. Elindeyse stop-loss'u sıkılaştır veya pozisyonu azalt.",
                "no_own": "Düşüş baskısı var; yeni giriş için dip ve hacim toparlanmasını bekle."
            }
        elif abs(ema_diff) < 0.5 and abs(obv_slope) < 1 and abs((recent_prices[-1] if recent_prices else 0) - (recent_prices[0] if recent_prices else 0)) / (recent_prices[0] if recent_prices else 1) * 100 < 1.5:
            trend_label = "⏸ Kararsız"
            advice_pair = {
                "own": "Piyasa kararsız. Yeni işlem açmadan önce hacim desteğini bekle.",
                "no_own": "Henüz net sinyal yok. RSI ve hacim yön değişimini bekle."
            }
        else:
            trend_label = "⚠️ Zayıflayan trend (güç: {}/10)".format(strength)
            advice_pair = {
                "own": "Momentum belirsiz; kârı korumak için stop belirle. Yeni alım yapma.",
                "no_own": "Trend kararsız. Fibo 38.2–61.8 aralığına geri dönüşü bekle."
            }

        return trend_label, advice_pair, strength

    def check_prices():
        STOP_LOSS_STATIC_PCT = 0.03  # her hisse için %3 statik stop-loss
        STOP_LOSS_RESET_PCT = 0.05   # %5 toparlanma sonrası stop tekrar aktifleşir
        STOP_LOSS_MARGIN = 0.01      # fibo destek altı marj (%1)

        tz = pytz.timezone(MARKET_TZ)
        now = datetime.now(tz)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] Fiyat kontrolü başlıyor...")

        if USE_MARKET_HOURS and not is_bist_open_now(now):
            print("⏸ Market kapalı (hafta sonu / tatil / saat dışında). Kontrol atlandı.")
            return

        for sym, meta in monitored_dict.items():
            latest = safe_get_last_price(sym)
            if latest is None:
                print(f"  {sym}: fiyat alınamadı.")
                continue

            baseline = meta.get("baseline_price")
            if baseline is None:
                print(f"  {sym}: baseline yok, atlandı.")
                continue

            meta.setdefault("fibo_alerts", {})
            meta.setdefault("alerts", {"balanced": False, "rsi": False})
            meta.setdefault("last_trend", None)
            meta.setdefault("last_price", baseline)
            meta.setdefault("stop_triggered", False)

            pct_from_baseline = (latest - baseline) / baseline * 100.0

            trend_data = get_trend_data(sym)
            if not trend_data:
                print(f"  {sym}: trend verisi yok.")
                continue
            rsi = trend_data.get("rsi")
            ema10 = trend_data.get("ema10")
            ema20 = trend_data.get("ema20")
            obv_slope = trend_data.get("obv_slope", 0.0)

            # --- küçük momentum kontrolü ---
            recent_prices = None
            try:
                t = yf.Ticker(f"{sym}.IS")
                hist = t.history(period="7d", interval="1d")
                if hist is not None and not hist.empty:
                    recent_prices = hist["Close"].dropna().tolist()
            except Exception:
                recent_prices = None

            # --- Fibonacci seviyeleri ---
            recent_low = baseline * 0.9
            recent_high = baseline * 1.1
            diff = (recent_high - recent_low) if (recent_high - recent_low) != 0 else 1.0
            fibo_levels = {
                23.6: recent_high - 0.236 * diff,
                38.2: recent_high - 0.382 * diff,
                50.0: recent_high - 0.500 * diff,
                61.8: recent_high - 0.618 * diff,
                78.6: recent_high - 0.786 * diff,
            }

            # --- Stop-Loss hesaplama ---
            static_stop = baseline * (1 - STOP_LOSS_STATIC_PCT)
            dynamic_stop = static_stop

            # fiyat son fibo desteğinin altına sarkarsa onu referans al
            for lvl, fib_price in sorted(fibo_levels.items(), reverse=True):
                if latest > fib_price:
                    dynamic_stop = fib_price * (1 - STOP_LOSS_MARGIN)
                    break

            stop_loss_price = min(static_stop, dynamic_stop)

            # stop-loss tetikleme kontrolü
            if not meta["stop_triggered"] and latest <= stop_loss_price:
                send_telegram_message(
                    f"🛑 {sym} STOP-LOSS Tetiklendi!\n"
                    f"Fiyat: {latest:.2f} ₺ ≤ Stop Seviyesi: {stop_loss_price:.2f} ₺\n"
                    f"💡 Tavsiye: Zararı büyütmemek için pozisyonu gözden geçir. Ana destek kırıldıysa çıkış değerlendir."
                )
                meta["stop_triggered"] = True

            # fiyat toparlanırsa stop resetlenir
            elif meta["stop_triggered"] and latest >= stop_loss_price * (1 + STOP_LOSS_RESET_PCT):
                meta["stop_triggered"] = False
                send_telegram_message(f"✅ {sym} fiyat toparlandı, stop-loss yeniden aktif hale getirildi ({latest:.2f} ₺)")

            # --- Fibonacci geçişleri ---
            fibo_crossed = []
            fibo_msgs = []
            for lvl, price_level in fibo_levels.items():
                key = f"fibo_{lvl}"
                if not meta["fibo_alerts"].get(key, False) and latest >= price_level:
                    meta["fibo_alerts"][key] = True
                    fibo_crossed.append(lvl)
                    # uygun tavsiyeler
                    if lvl == 23.6:
                        adv = "Trend yeni başlıyor olabilir. Küçük miktarda alım düşünülebilir; hacimle teyit bekle."
                    elif lvl == 38.2:
                        adv = "Güçlenme sinyali. OBV yükseliyorsa pozisyon korunabilir; RSI yüksekse kâr al."
                    elif lvl == 50.0:
                        adv = "Kısa vadeli momentum bölgesi. Tutunursa pozisyon artırılabilir."
                    elif lvl == 61.8:
                        adv = "Ana direnç. RSI orta seviyedeyse güçlü kırılma beklenir; RSI yüksekse kâr almayı düşün."
                    elif lvl == 78.6:
                        adv = "Yüksek direnç; düzeltme riski yükselir. Hacim düşükse çıkış düşün."
                    fibo_msgs.append(
                        f"\n\n📊 {sym} {lvl:.1f}% Fibonacci seviyesini geçti!"
                        f"\nSeviye: {price_level:.2f} ₺ | Güncel: {latest:.2f} ₺"
                        f"\n💡 Tavsiye: {adv}"
                    )

            # --- Trend analizi ---
            trend_label, advice_pair, trend_strength = analyze_trend_with_strength(
                rsi=rsi, ema10=ema10, ema20=ema20, obv_slope=obv_slope, recent_prices=recent_prices
            )

            last_trend = meta.get("last_trend")
            last_price = meta.get("last_price", baseline)
            trend_changed = (trend_label != last_trend)
            try:
                price_change_since_last = abs((latest - last_price) / (last_price if last_price else baseline) * 100.0)
            except Exception:
                price_change_since_last = 0.0
            big_move = price_change_since_last >= 3.0

            should_send = bool(fibo_crossed) or trend_changed or big_move

            # --- hedef fiyat alarmları ---
            for mkey, tkey, label in [
                ("balanced", "target_price_balanced", "Balanced"),
                ("rsi", "target_price_rsi", "RSI"),
            ]:
                tp = meta.get(tkey)
                if tp is not None and not meta["alerts"].get(mkey, False) and latest >= tp:
                    send_telegram_message(
                        f"🚨 {sym} {label} hedefe ulaştı!\n"
                        f"Şu an: {latest:.2f} ₺ \nHedef: {tp:.2f} ₺"
                    )
                    meta["alerts"][mkey] = True

            if should_send:
                parts = [
                    f"📊 {sym} Güncellemesi",
                    f"💰 Fiyat: {latest:.2f} ₺  ({pct_from_baseline:+.2f}% başlangıca göre)",
                    f"📈 Trend: {trend_label}",
                    f"💬 Eğer elinde VARSA: {advice_pair['own']}",
                    f"💬 Eğer elinde YOKSA: {advice_pair['no_own']}",
                ]
                if fibo_msgs:
                    parts.extend(fibo_msgs)
                if big_move:
                    parts.append(f"\n⚡ Büyük hareket: Son gönderime göre %{price_change_since_last:.2f} değişim.")
                parts.append(f"\n🔎 RSI:{rsi:.1f if rsi is not None else 'NA'} | EMA10-20 diff:{(ema10-ema20):.4f} | OBV_slope:{obv_slope:.2f} | Güç:{trend_strength}/10")
                parts.append(f"🧯 Stop-Loss: {stop_loss_price:.2f} ₺ (aktif)" if not meta["stop_triggered"] else f"🛑 Stop-Loss: {stop_loss_price:.2f} ₺ (tetiklendi)")
                msg = "\n".join(parts)
                send_telegram_message(msg)

                meta["last_trend"] = trend_label
                meta["last_price"] = latest

            print(f"  {sym}: trend={trend_label}, fiyat={latest:.2f} ₺, stop={stop_loss_price:.2f}, fibo_crossed={fibo_crossed}")

        print("✅ Kontrol tamamlandı.")

    return check_prices



# -----------------------------
# MAIN
# -----------------------------
def main():
    try:
        combined_df, monitored, top_dict = analyze_once()
    except Exception as e:
        print("Analiz sırasında hata:", e)
        return

    # monitored only with baseline price available
    monitored_valid = {s: m for s, m in monitored.items() if m.get("baseline_price") is not None}
    if not monitored_valid:
        print("Geçerli baseline fiyatı olan izlenecek sembol yok. Program sonlanıyor.")
        return

    print(f"\nİzlenen sembol sayısı: {len(monitored_valid)}")
    checker = create_price_checker(monitored_valid)

    # run initial check immediately (isteğe göre yorumlayabilirsin)
    checker()

    # schedule periodic checks
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(checker)
    print(f"İzleme başladı — her {CHECK_INTERVAL_MINUTES} dakikada bir kontrol edilecek. (USE_MARKET_HOURS={USE_MARKET_HOURS})")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program manuel olarak durduruldu.")
    except Exception as e:
        print("Ana döngüde hata:", e)


if __name__ == "__main__":
    main()

