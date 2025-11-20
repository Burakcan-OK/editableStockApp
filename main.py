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
# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
# Telegram (ortam değişkeni veya doğrudan buraya koyabilirsin)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", )
CHAT_ID = os.getenv("CHAT_ID", )

# Hareket bildirimi eşiği (örnek: 1.0 => %1)
MOVEMENT_NOTIFY_DWN = float(os.environ.get("MOVEMENT_NOTIFY_DWN", 1))
MOVEMENT_NOTIFY_UP = float(os.environ.get("MOVEMENT_NOTIFY_UP", 2.5))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", -3.5)) # %3.5 stop loss
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
bist = mcal.get_calendar("XIST")

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
print(f"  MOVEMENT_NOTIFY_DWN = {MOVEMENT_NOTIFY_DWN}%")
print(f"  MOVEMENT_NOTIFY_UP = {MOVEMENT_NOTIFY_UP}%")
print(f"  CHECK_INTERVAL_MINUTES = {CHECK_INTERVAL_MINUTES} minutes")
print(f"  USE_MARKET_HOURS = {USE_MARKET_HOURS}")
print(f"  MARKET HOURS = {MARKET_OPEN} -> {MARKET_CLOSE} ({MARKET_TZ})")
print(f"  Columns file = {COLUMNS_JSON}, Data file = {DATA_JSON}")
print("--------------------------------------------------\n")


# -----------------------------
# UTIL: Telegram & price fetch
# -----------------------------
load_dotenv()
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
    def check_prices():
        tz = pytz.timezone(MARKET_TZ)
        now = datetime.now(tz)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] Fiyat kontrolü başlıyor...")

        if USE_MARKET_HOURS and not is_bist_open_now(now):
            print("⏸ Market kapalı, kontrol atlandı.")
            return

        for sym, meta in monitored_dict.items():
            latest = safe_get_last_price(sym)
            if latest is None:
                print(f"  {sym}: fiyat alınamadı.")
                continue

            # --- Referans ayarları ---
            baseline = meta.get("baseline_price")
            if baseline is None:
                print(f"  {sym}: baseline yok, atlandı.")
                continue

            if "current_ref_price" not in meta or meta["current_ref_price"] is None:
                meta["current_ref_price"] = baseline

            ref_price = meta["current_ref_price"]

            pct_up = (latest - baseline) / baseline * 100.0          # baseline yükseliş
            pct_down = (latest - baseline) / baseline * 100.0        # baseline düşüş
            pct_local = (latest - ref_price) / ref_price * 100.0     # mini trend

            # -------------------------------------
            # 🟥 STOP-LOSS (baseline'a göre)
            # continue YOK → diğer kontroller çalışacak
            # -------------------------------------
            if pct_down <= STOP_LOSS_PCT:
                stop_steps = int(abs(pct_down) // abs(STOP_LOSS_PCT))
                last_steps = meta.get("stop_loss_steps", 0)

                if stop_steps > last_steps:
                    send_telegram_message(
                        f"🚨 {sym} STOP-LOSS ({stop_steps}. kademe)\n"
                        f"────────────────\n"
                        f"📆 {now.strftime('%Y-%m-%d %H:%M')}\n"
                        f"💰 Baseline: {baseline:.2f} ₺\n"
                        f"📉 Son fiyat: {latest:.2f} ₺\n"
                        f"🔻 Değişim: {pct_down:.2f}%\n"
                        f"⚠️ Stop-loss: {STOP_LOSS_PCT}% x {stop_steps}"
                    )
                    meta["stop_loss_steps"] = stop_steps

            else:
                # stop-loss seviyesinden yukarı çıkıldıysa resetle
                meta["stop_loss_steps"] = 0
            # -------------------------------------
            # 🟩 BASELINE YÜKSELİŞ
            # -------------------------------------
            if pct_up >= MOVEMENT_NOTIFY_UP:
                steps_up = int(pct_up // MOVEMENT_NOTIFY_UP)
                last_up = meta.get("last_base_up", 0)

                if steps_up > last_up:
                    send_telegram_message(
                        f"📈 {sym} yükseliş: +{pct_up:.2f}%\n"
                        f"💰 Baseline: {baseline:.2f} ₺\n"
                        f"📉 Son fiyat: {latest:.2f} ₺\n"
                    )
                    meta["last_base_up"] = steps_up
                    # mini trend tetiklenecek → yeni tepe referansı:
                    meta["current_ref_price"] = latest

            # -------------------------------------
            # 🟦 BASELINE DÜŞÜŞ
            # -------------------------------------
            if pct_down <= -MOVEMENT_NOTIFY_DWN:
                steps_dn = int(abs(pct_down) // MOVEMENT_NOTIFY_DWN)
                last_dn = meta.get("last_base_dn", 0)

                if steps_dn > last_dn:
                    send_telegram_message(
                        f"📉 {sym} düşüş: {pct_down:.2f}%\n"
                        f"💰 Baseline: {baseline:.2f} ₺\n"
                        f"📉 Son Fiyat: {latest:.2f} ₺\n"
                    )
                    meta["last_base_dn"] = steps_dn
                    # mini trend tetiklenecek → yeni dip referansı:
                    meta["current_ref_price"] = latest

            # -------------------------------------
            # 🟦 MINI TREND (Düşüş sonrası yükseliş)
            # -------------------------------------
            if pct_local >= MOVEMENT_NOTIFY_UP:
                steps_up = int(pct_local // MOVEMENT_NOTIFY_UP)
                last_up = meta.get("local_up", 0)

                if steps_up > last_up:
                    send_telegram_message(
                        f"📈 (Mini Trend) {sym} dipten yükseldi: +{pct_local:.2f}%\n"
                        f"💰 Referans Fiyat: {ref_price:.2f} ₺\n"
                        f"📉 Son Fiyat: {latest:.2f} ₺\n"
                    )
                    meta["local_up"] = steps_up

            # -------------------------------------
            # 🟥 MINI TREND (Yükseliş sonrası düşüş)
            # -------------------------------------
            if pct_local <= -MOVEMENT_NOTIFY_DWN:
                steps_dn = int(abs(pct_local) // MOVEMENT_NOTIFY_DWN)
                last_dn = meta.get("local_dn", 0)

                if steps_dn > last_dn:
                    send_telegram_message(
                        f"📉 (Mini Trend) {sym} tepeden düştü: {pct_local:.2f}%\n"
                        f"💰 Referans Fiyat: {ref_price:.2f} ₺\n"
                        f"📉 Son Fiyat: {latest:.2f} ₺\n"
                    )
                    meta["local_dn"] = steps_dn

            print(
                f"  {sym}: baseline={baseline:.2f}, latest={latest:.2f}, "
                f"pct_up={pct_up:.2f}%, pct_local={pct_local:.2f}%, ref={ref_price:.2f}"
            )

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

    monitored_valid = {s: m for s, m in monitored.items() if m.get("baseline_price") is not None}
    if not monitored_valid:
        print("Geçerli baseline fiyatı olan izlenecek sembol yok. Program sonlanıyor.")
        return

    print(f"\nİzlenen sembol sayısı: {len(monitored_valid)}")
    checker = create_price_checker(monitored_valid)

    # güvenli wrapper: market saatleri kontrolünü burada yapıyoruz,
    # böylece schedule tarafından çağrılsa bile wrapper içindeki kontroller devrede olur.
    def checker_wrapper():
        tz = pytz.timezone(MARKET_TZ)
        now = datetime.now(tz)

        if USE_MARKET_HOURS:
            # tatil / hafta sonu kontrolü
            if not is_bist_open_now(now):
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S %z')}] ⏸ Borsa kapalı (hafta sonu/tatil). Kontrol atlandı.")
                return

            # günlük market saatleri kontrolü (MARKET_OPEN, MARKET_CLOSE dtime objesi olmalı)
            if not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S %z')}] ⏸ Market saatleri dışında ({now.strftime('%H:%M:%S')}). Kontrol atlandı.")
                return

        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S %z')}] Fiyat kontrolü başlıyor...")
        try:
            checker()
        except Exception as e:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S %z')}] Checker hatası:", e)

    # schedule job (CHECK_INTERVAL_MINUTES'in int olduğundan emin ol)
    interval = int(CHECK_INTERVAL_MINUTES)
    schedule.every(interval).minutes.do(checker_wrapper)
    print(f"İzleme başladı — her {interval} dakikada bir kontrol edilecek. (USE_MARKET_HOURS={USE_MARKET_HOURS})")

    # İlk kontrolü yalnızca market açıksa çalıştır (veya USE_MARKET_HOURS False ise her zaman)
    tz = pytz.timezone(MARKET_TZ)
    now = datetime.now(tz)
    if (not USE_MARKET_HOURS) or (is_bist_open_now(now) and MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        checker_wrapper()
    else:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S %z')}] İlk kontrol atlandı (market kapalı).")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)  # hafif uyku; CPU kullanımı düşer
    except KeyboardInterrupt:
        print("Program manuel olarak durduruldu.")
    except Exception as e:
        print("Ana döngüde hata:", e)

if __name__ == "__main__":
    main()
