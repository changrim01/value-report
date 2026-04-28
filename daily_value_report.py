"""
daily_value_report.py
─────────────────────────────────────────────────────────────────────────────
Daily Value Investing Report – News Aggregation + Proxy DCF Screener
Author : Senior Quant Engineer (Warren Buffett / Charlie Munger philosophy)

Pipeline
  1. Fetch last-24h news headlines for watchlist tickers
  2. Run proxy-DCF screening across US + Japan ADR tickers
  3. Rank by safety-margin (margin of safety %) → top-5
  4. Compose mobile-friendly HTML e-mail & send via SMTP
─────────────────────────────────────────────────────────────────────────────
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import logging
import smtplib
import math
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Third-party ───────────────────────────────────────────────────────────────
import feedparser          # pip install feedparser
import yfinance as yf      # pip install yfinance
import pandas as pd        # pip install pandas

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (override via environment variables for GitHub Actions secrets)
# ─────────────────────────────────────────────────────────────────────────────

# E-mail credentials  →  set as GitHub Secrets: SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL
SENDER_EMAIL     = os.environ.get("SENDER_EMAIL",    "your_email@gmail.com")
SENDER_PASSWORD  = os.environ.get("SENDER_PASSWORD", "your_app_password")
RECIPIENT_EMAIL  = os.environ.get("RECIPIENT_EMAIL", "recipient@example.com")
SMTP_HOST        = os.environ.get("SMTP_HOST",        "smtp.gmail.com")
SMTP_PORT        = int(os.environ.get("SMTP_PORT",    "587"))

# Watchlist tickers (for news section)
NEWS_TICKERS = {
    "QCOM":   "Qualcomm",
    "GOOGL":  "Alphabet (Google)",
    "OKLO":   "Oklo",
    "8001.T": "Itochu",
    "8058.T": "Mitsubishi",
    "8031.T": "Mitsui",
    "8053.T": "Sumitomo",
    "8002.T": "Marubeni",
}

# Full screening universe for DCF (includes both US & Japan ADR tickers)
SCREENING_UNIVERSE = list(NEWS_TICKERS.keys())

# DCF hyper-parameters (Buffett-conservative)
TERMINAL_GROWTH_RATE = 0.025   # 2.5%
DISCOUNT_RATE        = 0.10    # 10%
MIN_ROIC_THRESHOLD   = 0.05    # 5%
NEWS_PER_TICKER      = 3
LOOKBACK_HOURS       = 72

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — NEWS AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def fetch_news_for_ticker(ticker: str, company_name: str, max_items: int = NEWS_PER_TICKER) -> list[dict]:
    """
    Fetch the last-24h news headlines for a single ticker via Google News RSS.

    Returns a list of dicts:
        [{"title": str, "link": str, "published": str}, ...]
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    # Use company name for better Google News recall
    query   = company_name.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"

    results = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            pub_time = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            if pub_time and pub_time < cutoff:
                continue   # skip articles older than 24h

            results.append({
                "title":     entry.get("title", "N/A"),
                "link":      entry.get("link",  "#"),
                "published": pub_time.strftime("%Y-%m-%d %H:%M UTC") if pub_time else "N/A",
            })

            if len(results) >= max_items:
                break

        log.info(f"[News] {ticker}: fetched {len(results)} item(s)")

    except Exception as exc:
        log.warning(f"[News] {ticker}: fetch failed → {exc}")

    return results


def aggregate_all_news(tickers: dict) -> dict:
    """Aggregate news for all tickers. Returns {ticker: [news_items]}."""
    all_news = {}
    for ticker, name in tickers.items():
        all_news[ticker] = fetch_news_for_ticker(ticker, name)
    return all_news


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — VALUE DISCOVERY / PROXY DCF SCREENER
# ═════════════════════════════════════════════════════════════════════════════

def _safe_get(obj, *keys, default=None):
    """Safely drill into nested yfinance data structures."""
    for key in keys:
        if obj is None:
            return default
        try:
            if isinstance(obj, pd.DataFrame):
                if key in obj.index:
                    val = obj.loc[key]
                    # Return most-recent column (first column after sort)
                    return val.iloc[0] if hasattr(val, "iloc") else val
                return default
            elif isinstance(obj, dict):
                obj = obj.get(key, default)
            else:
                return default
        except Exception:
            return default
    return obj if obj is not None else default


def calculate_roic(info: dict, cashflow: pd.DataFrame, balance: pd.DataFrame) -> float | None:
    """
    ROIC proxy = Operating Cash Flow / (Total Assets - Current Liabilities)
    Falls back to Return on Assets if balance sheet data is incomplete.
    """
    try:
        ocf = _safe_get(cashflow, "Operating Cash Flow")
        if ocf is None or not math.isfinite(float(ocf)):
            return None

        total_assets      = float(info.get("totalAssets", 0) or 0)
        current_liab      = float(info.get("totalCurrentLiabilities", 0) or 0)
        invested_capital  = total_assets - current_liab

        if invested_capital <= 0:
            return None

        return float(ocf) / invested_capital

    except Exception:
        return None


def calculate_intrinsic_value(fcf: float, shares_outstanding: float) -> float | None:
    """
    Buffett-style simplified terminal value (Gordon Growth Model):

        IV_per_share = FCF × (1 + g) / (r − g)  /  shares_outstanding

    where
        FCF  = most-recent Free Cash Flow (total, not per-share)
        g    = TERMINAL_GROWTH_RATE  (2.5%)
        r    = DISCOUNT_RATE         (10%)
    """
    if fcf is None or fcf <= 0 or shares_outstanding <= 0:
        return None

    try:
        terminal_value   = fcf * (1 + TERMINAL_GROWTH_RATE) / (DISCOUNT_RATE - TERMINAL_GROWTH_RATE)
        iv_per_share     = terminal_value / shares_outstanding
        return round(iv_per_share, 2)
    except ZeroDivisionError:
        return None


def analyse_ticker(ticker: str) -> dict | None:
    """
    Pull yfinance data and compute proxy-DCF metrics for a single ticker.

    Returns a dict or None if data is insufficient / company fails filters.
    """
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info or {}
        cashflow = stock.cashflow          # DataFrame, columns = fiscal-year dates
        balance  = stock.balance_sheet

        # ── Basic sanity checks ───────────────────────────────────────────
        current_price    = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        shares_out       = float(info.get("sharesOutstanding") or 0)
        net_income       = float(info.get("netIncomeToCommon") or 0)

        if current_price <= 0 or shares_out <= 0:
            log.warning(f"[DCF] {ticker}: missing price or shares data, skipping.")
            return None

        # ── Filter: must be profitable ────────────────────────────────────
        if net_income <= 0:
            log.info(f"[DCF] {ticker}: net income ≤ 0, skipping (not yet profitable).")
            return None

        # ── FCF calculation ───────────────────────────────────────────────
        ocf = _safe_get(cashflow, "Operating Cash Flow")
        cap = _safe_get(cashflow, "Capital Expenditure")

        if ocf is None:
            log.warning(f"[DCF] {ticker}: operating cash flow not found, skipping.")
            return None

        ocf  = float(ocf)
        capex = float(cap) if cap is not None else 0.0
        # CapEx is usually reported as a negative number in yfinance
        fcf  = ocf + capex if capex < 0 else ocf - capex

        if fcf <= 0:
            log.info(f"[DCF] {ticker}: FCF ≤ 0, noted.")
            #return None

        # ── ROIC filter ───────────────────────────────────────────────────
        roic = calculate_roic(info, cashflow, balance)
        if roic is None or roic < MIN_ROIC_THRESHOLD:
            log.info(f"[DCF] {ticker}: ROIC {roic} below threshold, skipping.")
            return None

        # ── Intrinsic value & safety margin ──────────────────────────────
        iv = calculate_intrinsic_value(fcf, shares_out)
        if iv is None or iv <= 0:
            log.warning(f"[DCF] {ticker}: intrinsic value calc failed, skipping.")
            return None

        safety_margin = (iv - current_price) / iv  # positive → undervalued

        currency = info.get("currency", "USD")

        return {
            "ticker":         ticker,
            "company":        info.get("longName") or ticker,
            "current_price":  round(current_price, 2),
            "intrinsic_value": iv,
            "safety_margin":  round(safety_margin * 100, 1),   # in %
            "fcf_bn":         round(fcf / 1e9, 2),             # in billions
            "roic_pct":       round(roic * 100, 1),
            "currency":       currency,
        }

    except Exception as exc:
        log.warning(f"[DCF] {ticker}: unexpected error → {exc}")
        return None


def screen_universe(tickers: list[str], top_n: int = 5) -> pd.DataFrame:
    """
    Screen all tickers, apply filters, rank by safety margin, return top-N.
    """
    log.info(f"[DCF] Screening {len(tickers)} ticker(s) …")
    records = []
    for t in tickers:
        result = analyse_ticker(t)
        if result:
            records.append(result)
            log.info(
                f"[DCF] {t:6s} | Price={result['current_price']:>8.2f}"
                f" | IV={result['intrinsic_value']:>8.2f}"
                f" | MoS={result['safety_margin']:>6.1f}%"
                f" | ROIC={result['roic_pct']:>5.1f}%"
            )

    if not records:
        log.warning("[DCF] No qualifying stocks found in screening pass.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("safety_margin", ascending=False).head(top_n)
    df.reset_index(drop=True, inplace=True)
    df.index += 1   # 1-based ranking
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — HTML REPORT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
  /* ── Reset & base ── */
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    padding: 16px;
  }
  .wrapper   { max-width: 680px; margin: 0 auto; }
  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: #e0e0e0;
    border-radius: 12px 12px 0 0;
    padding: 28px 24px 20px;
  }
  .header h1  { font-size: 1.45rem; letter-spacing: .4px; color: #fff; }
  .header p   { font-size: .82rem; color: #a0aec0; margin-top: 4px; }
  .badge {
    display: inline-block;
    background: #e94560;
    color: #fff;
    font-size: .7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 8px;
    vertical-align: middle;
  }
  /* ── Sections ── */
  .section {
    background: #fff;
    padding: 22px 20px;
    border-left: 4px solid #0f3460;
    margin-top: 2px;
  }
  .section:last-child { border-radius: 0 0 12px 12px; }
  .section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0f3460;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e8edf2;
  }
  /* ── DCF Table ── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: .83rem;
  }
  th {
    background: #1a1a2e;
    color: #a0c4ff;
    padding: 9px 10px;
    text-align: left;
    font-weight: 600;
  }
  td { padding: 8px 10px; border-bottom: 1px solid #f0f2f5; }
  tr:hover td { background: #f7f9fc; }
  .rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    font-size: .72rem;
    font-weight: 700;
    color: #fff;
  }
  .r1 { background: #f6c90e; color: #1a1a2e; }
  .r2 { background: #9aa5b1; }
  .r3 { background: #cd7f32; }
  .r45{ background: #3a7bd5; }
  .mos-positive { color: #27ae60; font-weight: 700; }
  .mos-negative { color: #e94560; font-weight: 700; }
  /* ── News grid ── */
  .news-company { font-weight: 700; color: #0f3460; margin-top: 14px; font-size: .9rem; }
  .news-item { margin: 5px 0 5px 10px; font-size: .82rem; }
  .news-item a {
    color: #3a7bd5;
    text-decoration: none;
    line-height: 1.5;
  }
  .news-item a:hover { text-decoration: underline; }
  .news-meta { font-size: .72rem; color: #9aa5b1; margin-left: 10px; }
  /* ── Footer ── */
  .footer {
    font-size: .72rem;
    color: #9aa5b1;
    text-align: center;
    padding: 14px 0 4px;
    line-height: 1.6;
  }
  /* ── Mobile ── */
  @media (max-width: 480px) {
    body { padding: 8px; }
    .header { padding: 18px 14px 14px; }
    table { font-size: .76rem; }
    th, td { padding: 7px 7px; }
  }
</style>
"""

RANK_CLASS = {1: "r1", 2: "r2", 3: "r3", 4: "r45", 5: "r45"}


def _build_dcf_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#e94560'>⚠️ No qualifying stocks passed the screening criteria today.</p>"

    rows = ""
    for rank, row in df.iterrows():
        badge_cls = RANK_CLASS.get(rank, "r45")
        mos_cls   = "mos-positive" if row["safety_margin"] > 0 else "mos-negative"
        mos_sign  = "+" if row["safety_margin"] > 0 else ""
        rows += f"""
        <tr>
          <td><span class="rank-badge {badge_cls}">{rank}</span></td>
          <td><strong>{row['ticker']}</strong><br>
              <span style="font-size:.76rem;color:#666">{row['company'][:28]}</span></td>
          <td>{row['currency']} {row['current_price']:,.2f}</td>
          <td>{row['currency']} {row['intrinsic_value']:,.2f}</td>
          <td class="{mos_cls}">{mos_sign}{row['safety_margin']:.1f}%</td>
          <td>{row['fcf_bn']:.2f}B</td>
          <td>{row['roic_pct']:.1f}%</td>
        </tr>"""

    return f"""
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker / Name</th><th>Current Price</th>
          <th>Intrinsic Value*</th><th>Margin of Safety</th>
          <th>FCF (ann.)</th><th>ROIC</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <p style="font-size:.72rem;color:#9aa5b1;margin-top:8px">
      * Proxy IV = FCF × (1+g)/(r-g) ÷ shares outstanding &nbsp;|&nbsp;
        g={TERMINAL_GROWTH_RATE*100:.1f}%&nbsp; r={DISCOUNT_RATE*100:.0f}%
    </p>"""


def _build_news_section(all_news: dict) -> str:
    html = ""
    for ticker, news_list in all_news.items():
        company = NEWS_TICKERS.get(ticker, ticker)
        html += f'<div class="news-company">📰 {company} ({ticker})</div>'
        if not news_list:
            html += '<p class="news-item" style="color:#9aa5b1">No recent news found.</p>'
        else:
            for item in news_list:
                html += f"""
                <div class="news-item">
                  • <a href="{item['link']}" target="_blank">{item['title']}</a>
                </div>
                <div class="news-meta">{item['published']}</div>"""
    return html


def build_html_email(dcf_df: pd.DataFrame, all_news: dict) -> str:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M KST")
    dcf_html  = _build_dcf_table(dcf_df)
    news_html = _build_news_section(all_news)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Value Report</title>
{_CSS}
</head>
<body>
<div class="wrapper">

  <!-- ── HEADER ── -->
  <div class="header">
    <h1>📊 Daily Value Investing Report
      <span class="badge">AUTO</span>
    </h1>
    <p>Warren Buffett–style Proxy DCF Screener &nbsp;|&nbsp; {run_time}</p>
  </div>

  <!-- ── DCF TOP-5 ── -->
  <div class="section">
    <div class="section-title">🏆 Top-5 Value Picks (Proxy DCF Screening)</div>
    {dcf_html}
  </div>

  <!-- ── WATCHLIST NEWS ── -->
  <div class="section">
    <div class="section-title">🗞️ Watchlist News (Last 24 Hours)</div>
    {news_html}
  </div>

  <!-- ── DISCLAIMER ── -->
  <div class="section" style="border-left-color:#e94560; background:#fff9f9;">
    <div class="section-title" style="color:#e94560;">⚠️ Disclaimer</div>
    <p style="font-size:.79rem; line-height:1.7; color:#555;">
      This report is generated automatically for <strong>personal research purposes only</strong>
      and does <u>not</u> constitute financial advice. The Proxy DCF model is a simplified
      heuristic based on trailing FCF and static growth assumptions — it does <u>not</u>
      account for debt, dilution, currency risk, or qualitative moats. Always conduct
      independent due diligence before making any investment decision.
    </p>
  </div>

</div><!-- /wrapper -->
<div class="footer">
  Generated by daily_value_report.py &nbsp;·&nbsp;
  Universe: {', '.join(SCREENING_UNIVERSE)} &nbsp;·&nbsp;
  Powered by yfinance &amp; Google News RSS
</div>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — EMAIL DELIVERY
# ═════════════════════════════════════════════════════════════════════════════

def send_html_email(subject: str, html_body: str) -> bool:
    """
    Send an HTML e-mail via SMTP (TLS).
    Returns True on success, False on failure.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL

    # Attach plain-text fallback
    plain = "This e-mail requires an HTML-capable mail client."
    msg.attach(MIMEText(plain,    "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html",  "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        log.info(f"[Mail] Report sent → {RECIPIENT_EMAIL}")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("[Mail] SMTP authentication failed. Check SENDER_EMAIL / SENDER_PASSWORD.")
    except smtplib.SMTPException as exc:
        log.error(f"[Mail] SMTP error: {exc}")
    except Exception as exc:
        log.error(f"[Mail] Unexpected error: {exc}")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_date = datetime.now().strftime("%Y-%m-%d")
    log.info("═" * 60)
    log.info(f"  Daily Value Report — {run_date}")
    log.info("═" * 60)

    # ── Step 1: News aggregation ─────────────────────────────────────────────
    log.info("[Step 1/4] Aggregating news …")
    all_news = aggregate_all_news(NEWS_TICKERS)

    # ── Step 2: DCF screening ────────────────────────────────────────────────
    log.info("[Step 2/4] Running Proxy-DCF screener …")
    dcf_results = screen_universe(SCREENING_UNIVERSE, top_n=5)

    if not dcf_results.empty:
        log.info("\n── Top-5 Value Picks ──────────────────────────────────────")
        log.info(dcf_results[["ticker", "current_price", "intrinsic_value",
                               "safety_margin", "roic_pct"]].to_string())
        log.info("──────────────────────────────────────────────────────────\n")
    else:
        log.warning("[Step 2/4] DCF screener returned no results.")

    # ── Step 3: Build HTML report ────────────────────────────────────────────
    log.info("[Step 3/4] Building HTML report …")
    html_report = build_html_email(dcf_results, all_news)

    # Optionally save a local copy for debugging
    local_path = f"report_{run_date}.html"
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        log.info(f"[Step 3/4] HTML report saved locally → {local_path}")
    except IOError as exc:
        log.warning(f"[Step 3/4] Could not save local report: {exc}")

    # ── Step 4: Send e-mail ──────────────────────────────────────────────────
    log.info("[Step 4/4] Sending e-mail …")
    subject = f"📊 Daily Value Report — {run_date} | Top-5 Proxy DCF Picks"
    success = send_html_email(subject, html_report)

    if success:
        log.info("✅ Pipeline completed successfully.")
    else:
        log.error("❌ Pipeline completed with e-mail delivery failure.")
