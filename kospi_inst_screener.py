"""
kospi_inst_screener.py
──────────────────────────────────────────────────────────────────────────────
KOSPI Institutional Buying Screener

Conditions (over last 10 KOSPI business days):
  1. Institutional net buying (기관합계 순매수) > 0
  2. Price return (start close → end close) < 5%

Output: top-5 tickers sorted by last business day trading volume

Install: pip install pykrx pandas
──────────────────────────────────────────────────────────────────────────────
"""

import sys
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS  = 10    # business days
MAX_RETURN_PCT = 5.0   # filter: keep tickers with return < this value
TOP_N          = 5     # how many results to display


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_last_n_business_days(n: int = LOOKBACK_DAYS) -> list[str]:
    """Return last n KOSPI business days as 'YYYYMMDD' strings."""
    today = datetime.now()
    # Look back far enough to guarantee n business days (account for holidays)
    look_back = (today - timedelta(days=n * 3)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    # Use KOSPI composite index (코드 1001) as the business-day calendar
    df = stock.get_index_ohlcv_by_date(look_back, end, "1001")
    if df.empty:
        raise RuntimeError("KOSPI 캘린더 데이터를 가져올 수 없습니다.")

    dates = df.index.strftime("%Y%m%d").tolist()
    if len(dates) < n:
        raise RuntimeError(f"영업일 데이터가 부족합니다 (확인된 일수: {len(dates)}).")
    return dates[-n:]


def fetch_institutional_net_buying(start: str, end: str) -> pd.Series:
    """
    Return a Series (index=ticker, values=기관합계 순매수 in KRW) for all
    KOSPI tickers over [start, end].  Positive = net buying.
    """
    df = stock.get_market_trading_value_by_ticker(start, end, market="KOSPI")
    if df.empty:
        raise RuntimeError("기관 매매 데이터를 가져올 수 없습니다.")

    # pykrx returns a column named '기관합계' for institutional aggregate
    if "기관합계" not in df.columns:
        raise KeyError(
            f"'기관합계' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}"
        )
    return df["기관합계"]


def fetch_ohlcv_snapshot(date: str) -> pd.DataFrame:
    """Return OHLCV DataFrame for all KOSPI tickers on a single date."""
    df = stock.get_market_ohlcv_by_ticker(date, market="KOSPI")
    if df.empty:
        raise RuntimeError(f"{date} 시세 데이터를 가져올 수 없습니다.")
    return df


def get_ticker_name(ticker: str) -> str:
    try:
        return stock.get_market_ticker_name(ticker)
    except Exception:
        return ticker


# ─────────────────────────────────────────────────────────────────────────────
# SCREENING
# ─────────────────────────────────────────────────────────────────────────────

def run_screener() -> pd.DataFrame:
    # ── Step 1: business-day calendar ────────────────────────────────────────
    print("  [1/4] 영업일 계산 중 …")
    days = get_last_n_business_days(LOOKBACK_DAYS)
    start_date, end_date = days[0], days[-1]
    print(f"        기간: {start_date} ~ {end_date}  ({len(days)} 영업일)")

    # ── Step 2: institutional net buying ──────────────────────────────────────
    print("  [2/4] 기관 순매수 데이터 수집 중 …")
    inst = fetch_institutional_net_buying(start_date, end_date)
    inst_positive = inst[inst > 0]
    print(f"        기관 순매수 > 0 종목 수: {len(inst_positive)}")

    if inst_positive.empty:
        print("  ⚠️  기관 순매수 종목이 없습니다.")
        return pd.DataFrame()

    # ── Step 3: price return filter ───────────────────────────────────────────
    print("  [3/4] 수익률 계산 중 …")
    start_ohlcv = fetch_ohlcv_snapshot(start_date)
    end_ohlcv   = fetch_ohlcv_snapshot(end_date)

    common = (
        inst_positive.index
        .intersection(start_ohlcv.index)
        .intersection(end_ohlcv.index)
    )

    start_close = start_ohlcv.loc[common, "종가"]
    end_close   = end_ohlcv.loc[common, "종가"]

    # Guard against zero start price
    valid = start_close[start_close > 0].index
    common = common.intersection(valid)

    returns_pct = ((end_ohlcv.loc[common, "종가"] - start_ohlcv.loc[common, "종가"])
                   / start_ohlcv.loc[common, "종가"] * 100).round(2)

    qualified = returns_pct[returns_pct < MAX_RETURN_PCT]
    print(f"        수익률 < {MAX_RETURN_PCT}% 조건 통과 종목 수: {len(qualified)}")

    if qualified.empty:
        print("  ⚠️  조건을 만족하는 종목이 없습니다.")
        return pd.DataFrame()

    # ── Step 4: collect metrics, sort by last-day volume ─────────────────────
    print("  [4/4] 결과 정렬 중 …")
    tickers = qualified.index.tolist()

    df = pd.DataFrame(
        {
            "종목명":        [get_ticker_name(t) for t in tickers],
            "현재가(원)":    end_ohlcv.loc[tickers, "종가"].values,
            "수익률(%)":     qualified.loc[tickers].values,
            "거래량(주)":    end_ohlcv.loc[tickers, "거래량"].values,
            "기관순매수(원)": inst_positive.loc[tickers].values,
        },
        index=tickers,
    )

    df = df.sort_values("거래량(주)", ascending=False).head(TOP_N)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_result(df: pd.DataFrame) -> None:
    SEP = "=" * 80
    print(f"\n{SEP}")
    print(
        f"  🏆 결과 : 기관 순매수 & 수익률 {MAX_RETURN_PCT}% 미만"
        f" — 거래량 상위 {TOP_N} 종목"
    )
    print(f"{SEP}")

    if df.empty:
        print("  조건을 만족하는 종목이 없습니다.")
        print(SEP)
        return

    fmt = df.copy()
    fmt["현재가(원)"]     = fmt["현재가(원)"].apply(lambda x: f"{x:>10,.0f}")
    fmt["수익률(%)"]      = fmt["수익률(%)"].apply(lambda x: f"{x:>8.2f}")
    fmt["거래량(주)"]     = fmt["거래량(주)"].apply(lambda x: f"{x:>14,.0f}")
    fmt["기관순매수(원)"]  = fmt["기관순매수(원)"].apply(lambda x: f"{x:>20,.0f}")

    print(fmt.to_string())
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  KOSPI 기관 순매수 스크리너")
    print("=" * 80)

    try:
        result_df = run_screener()
        print_result(result_df)
    except Exception as exc:
        print(f"\n❌ 오류 발생: {exc}", file=sys.stderr)
        sys.exit(1)
