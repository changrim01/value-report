"""
daily_value_report.py
"""

import os
import logging
import smtplib
import math
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import feedparser
import yfinance as yf
import pandas as pd

SENDER_EMAIL     = os.environ.get("SENDER_EMAIL",    "your_email@gmail.com")
SENDER_PASSWORD  = os.environ.get("SENDER_PASSWORD", "your_app_password")
RECIPIENT_EMAIL  = os.environ.get("RECIPIENT_EMAIL", "recipient@example.com")
SMTP_HOST        = os.environ.get("SMTP_HOST",       "smtp.gmail.com")
SMTP_PORT        = int(os.environ.get("SMTP_PORT",   "587"))

NEWS_TICKERS = {
    "QCOM":   "Qualcomm",
    "GOOGL":  "Alphabet Google",
    "OKLO":   "Oklo nuclear",
    "8001.T": "Itochu",
    "8058.T": "Mitsubishi Corporation",
    "8031.T": "Mitsui Co",
    "8053.T": "Sumitomo Corporation",
    "8002.T": "Marubeni",
}

SCREENING_UNIVERSE   = list(NEWS_TICKERS.keys())
TERMINAL_GROWTH_RATE = 0.025
DISCOUNT_RATE        = 0.10
NEWS_PER_TICKER      = 3
LOOKBACK_HOURS       = 72

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def fetch_news_for_ticker(ticker, company_name, max_items=NEWS_PER_TICKER):
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    query   = company_name.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    results = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            pub_time = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if pub_time and pub_time < cutoff:
                continue
            results.append({
                "title":     entry.get("title", "N/A"),
                "link":      entry.get("link",  "#"),
                "published": pub_time.strftime("%Y-%m-%d %H:%M UTC") if pub_time else "N/A",
            })
            if len(results) >= max_items:
                break
        log.info(f"[News] {ticker}: fetched {len(results)} item(s)")
    except Exception as exc:
        log.warning(f"[News] {ticker}: fetch failed -> {exc}")
    return results


def aggregate_all_news(tickers):
    return {ticker: fetch_news_for_ticker(ticker, name) for ticker, name in tickers.items()}


def _safe_get(obj, *keys, default=None):
    for key in keys:
        if obj is None:
            return default
        try:
            if isinstance(obj, pd.DataFrame):
                if key in obj.index:
                    val = obj.loc[key]
                    return val.iloc[0] if hasattr(val, "iloc") else val
                return default
            elif isinstance(obj, dict):
                obj = obj.get(key, default)
            else:
                return default
        except Exception:
            return default
    return obj if obj is not None else default


def calculate_intrinsic_value(fcf, shares_outstanding):
    # FCF <= 0 조건 제거 — 음수 FCF도 계산 수행
    if fcf is None or shares_outstanding <= 0:
        return None
    try:
        terminal_value = fcf * (1 + TERMINAL_GROWTH_RATE) / (DISCOUNT_RATE - TERMINAL_GROWTH_RATE)
        return round(terminal_value / shares_outstanding, 2)
    except ZeroDivisionError:
        return None


def analyse_ticker(ticker):
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info or {}
        cashflow = stock.cashflow

        current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        shares_out    = float(info.get("sharesOutstanding") or 0)

        if current_price <= 0 or shares_out <= 0:
            log.warning(f"[DCF] {ticker}: 가격/주식수 없음, skip")
            return None

        ocf = _safe_get(cashflow, "Operating Cash Flow")
        cap = _safe_get(cashflow, "Capital Expenditure")

        if ocf is None:
            log.warning(f"[DCF] {ticker}: OCF 없음, skip")
            return None

        ocf   = float(ocf)
        capex = float(cap) if cap is not None else 0.0
        fcf   = ocf + capex if capex < 0 else ocf - capex

        log.info(f"[DCF] {ticker}: OCF={ocf/1e9:.2f}B  CapEx={capex/1e9:.2f}B  FCF={fcf/1e9:.2f}B")

        roic = 0.0
        try:
            total_assets = float(info.get("totalAssets", 0) or 0)
            current_liab = float(info.get("totalCurrentLiabilities", 0) or 0)
            invested_cap = total_assets - current_liab
            if invested_cap > 0 and math.isfinite(ocf):
                roic = ocf / invested_cap
        except Exception:
            pass

        iv = calculate_intrinsic_value(fcf, shares_out)
        if iv is None:
            log.warning(f"[DCF] {ticker}: 내재가치 계산 실패, skip")
            return None

        safety_margin = (iv - current_price) / iv if iv != 0 else 0
        currency      = info.get("currency", "USD")

        log.info(f"[DCF] {ticker:6s} | Price={current_price:>8.2f} | IV={iv:>8.2f} | MoS={safety_margin*100:>6.1f}% | ROIC={roic*100:>5.1f}%")

        return {
            "ticker":          ticker,
            "company":         info.get("longName") or ticker,
            "current_price":   round(current_price, 2),
            "intrinsic_value": iv,
            "safety_margin":   round(safety_margin * 100, 1),
            "fcf_bn":          round(fcf / 1e9, 2),
            "roic_pct":        round(roic * 100, 1),
            "currency":        currency,
        }

    except Exception as exc:
        log.warning(f"[DCF] {ticker}: 예외 -> {exc}")
        return None


def screen_universe(tickers, top_n=5):
    log.info(f"[DCF] {len(tickers)}개 종목 스크리닝 …")
    records = [r for t in tickers if (r := analyse_ticker(t)) is not None]
    if not records:
        log.warning("[DCF] 통과 종목 없음")
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("safety_margin", ascending=False).head(top_n)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df


_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 16px; }
  .wrapper { max-width: 680px; margin: 0 auto; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); color: #e0e0e0; border-radius: 12px 12px 0 0; padding: 28px 24px 20px; }
  .header h1 { font-size: 1.45rem; letter-spacing: .4px; color: #fff; }
  .header p  { font-size: .82rem; color: #a0aec0; margin-top: 4px; }
  .badge { display: inline-block; background: #e94560; color: #fff; font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-left: 8px; vertical-align: middle; }
  .section { background: #fff; padding: 22px 20px; border-left: 4px solid #0f3460; margin-top: 2px; }
  .section:last-child { border-radius: 0 0 12px 12px; }
  .section-title { font-size: 1rem; font-weight: 700; color: #0f3460; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid #e8edf2; }
  table { width: 100%; border-collapse: collapse; font-size: .83rem; }
  th { background: #1a1a2e; color: #a0c4ff; padding: 9px 10px; text-align: left; font-weight: 600; }
  td { padding: 8px 10px; border-bottom: 1px solid #f0f2f5; }
  tr:hover td { background: #f7f9fc; }
  .rank-badge { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; font-size: .72rem; font-weight: 700; color: #fff; }
  .r1 { background: #f6c90e; color: #1a1a2e; } .r2 { background: #9aa5b1; } .r3 { background: #cd7f32; } .r45 { background: #3a7bd5; }
  .mos-positive { color: #27ae60; font-weight: 700; } .mos-negative { color: #e94560; font-weight: 700; }
  .news-company { font-weight: 700; color: #0f3460; margin-top: 14px; font-size: .9rem; }
  .news-item { margin: 5px 0 5px 10px; font-size: .82rem; }
  .news-item a { color: #3a7bd5; text-decoration: none; line-height: 1.5; }
  .news-meta { font-size: .72rem; color: #9aa5b1; margin-left: 10px; }
  .footer { font-size: .72rem; color: #9aa5b1; text-align: center; padding: 14px 0 4px; line-height: 1.6; }
  @media (max-width: 480px) { body { padding: 8px; } table { font-size: .76rem; } th, td { padding: 7px 7px; } }
</style>
"""

RANK_CLASS = {1: "r1", 2: "r2", 3: "r3", 4: "r45", 5: "r45"}


def _build_dcf_table(df):
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
          <td><strong>{row['ticker']}</strong><br><span style="font-size:.76rem;color:#666">{row['company'][:28]}</span></td>
          <td>{row['currency']} {row['current_price']:,.2f}</td>
          <td>{row['currency']} {row['intrinsic_value']:,.2f}</td>
          <td class="{mos_cls}">{mos_sign}{row['safety_margin']:.1f}%</td>
          <td>{row['fcf_bn']:.2f}B</td>
          <td>{row['roic_pct']:.1f}%</td>
        </tr>"""
    return f"""
    <table>
      <thead><tr><th>#</th><th>Ticker / Name</th><th>Current Price</th><th>Intrinsic Value*</th><th>Margin of Safety</th><th>FCF (ann.)</th><th>ROIC</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p style="font-size:.72rem;color:#9aa5b1;margin-top:8px">* Proxy IV = FCF×(1+g)/(r-g)÷shares &nbsp;|&nbsp; g={TERMINAL_GROWTH_RATE*100:.1f}%  r={DISCOUNT_RATE*100:.0f}%</p>"""


def _build_news_section(all_news):
    html = ""
    for ticker, news_list in all_news.items():
        company = NEWS_TICKERS.get(ticker, ticker)
        html += f'<div class="news-company">📰 {company} ({ticker})</div>'
        if not news_list:
            html += '<p class="news-item" style="color:#9aa5b1">No recent news found.</p>'
        else:
            for item in news_list:
                html += f'<div class="news-item">• <a href="{item["link"]}" target="_blank">{item["title"]}</a></div><div class="news-meta">{item["published"]}</div>'
    return html


def build_html_email(dcf_df, all_news):
    run_time  = datetime.now().strftime("%Y-%m-%d %H:%M KST")
    dcf_html  = _build_dcf_table(dcf_df)
    news_html = _build_news_section(all_news)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Daily Value Report</title>{_CSS}</head>
<body><div class="wrapper">
  <div class="header"><h1>📊 Daily Value Investing Report <span class="badge">AUTO</span></h1><p>Warren Buffett–style Proxy DCF Screener &nbsp;|&nbsp; {run_time}</p></div>
  <div class="section"><div class="section-title">🏆 Top-5 Value Picks (Proxy DCF Screening)</div>{dcf_html}</div>
  <div class="section"><div class="section-title">🗞️ Watchlist News (Last {LOOKBACK_HOURS}h)</div>{news_html}</div>
  <div class="section" style="border-left-color:#e94560;background:#fff9f9;"><div class="section-title" style="color:#e94560;">⚠️ Disclaimer</div>
  <p style="font-size:.79rem;line-height:1.7;color:#555;">Personal research only. Not financial advice. Always conduct independent due diligence.</p></div>
</div>
<div class="footer">daily_value_report.py &nbsp;·&nbsp; {', '.join(SCREENING_UNIVERSE)} &nbsp;·&nbsp; yfinance + Google News RSS</div>
</body></html>"""


def send_html_email(subject, html_body):
    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText("HTML-capable mail client required.", "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo(); server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        log.info(f"[Mail] 발송 완료 -> {RECIPIENT_EMAIL}")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("[Mail] 인증 실패")
    except Exception as exc:
        log.error(f"[Mail] 오류: {exc}")
    return False


if __name__ == "__main__":
    run_date = datetime.now().strftime("%Y-%m-%d")
    log.info("=" * 60)
    log.info(f"  Daily Value Report — {run_date}")
    log.info("=" * 60)

    log.info("[1/4] 뉴스 수집 중 …")
    all_news = aggregate_all_news(NEWS_TICKERS)

    log.info("[2/4] DCF 스크리닝 중 …")
    dcf_results = screen_universe(SCREENING_UNIVERSE, top_n=5)

    if not dcf_results.empty:
        log.info(dcf_results[["ticker", "current_price", "intrinsic_value", "safety_margin", "roic_pct"]].to_string())
    else:
        log.warning("[2/4] 통과 종목 없음")

    log.info("[3/4] HTML 리포트 생성 중 …")
    html_report = build_html_email(dcf_results, all_news)
    local_path  = f"report_{run_date}.html"
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        log.info(f"[3/4] 저장 완료 -> {local_path}")
    except IOError as exc:
        log.warning(f"[3/4] 저장 실패: {exc}")

    log.info("[4/4] 이메일 발송 중 …")
    success = send_html_email(f"📊 Daily Value Report — {run_date} | Top-5 Proxy DCF Picks", html_report)
    log.info("✅ 완료" if success else "❌ 이메일 발송 실패")
