import os
from datetime import datetime, date, time as dtime

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame


# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="WhisperChart — Intraday (Alpaca)", layout="wide")
st.title("📈 WhisperChart — Intraday (Alpaca)")

# Prefer Streamlit secrets; fall back to env vars for local dev
API_KEY = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))

if not API_KEY or not SECRET_KEY:
    st.warning("⚠️ Alpaca API keys missing. Add them to `.streamlit/secrets.toml` or export env vars.")

# Cache the client so we don't re-init each refresh
@st.cache_resource(show_spinner=False)
def get_client():
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

client = get_client()
EASTERN = pytz.timezone("US/Eastern")


# ====================== HELPERS ======================
def today_open_close_et():
    """Today's regular session times as ET-aware datetimes."""
    open_et = EASTERN.localize(datetime.combine(date.today(), dtime(9, 30)))
    close_et = EASTERN.localize(datetime.combine(date.today(), dtime(16, 0)))
    return open_et, close_et


def fetch_intraday_bars(symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
    """Bars from today's open (ET) to now (ET). Returns tidy DataFrame."""
    open_et, _ = today_open_close_et()
    now_et = datetime.now(EASTERN)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=open_et,
        end=now_et,
        adjustment=None,   # raw
        feed="sip",        # best quality if available on your account; Alpaca may fallback
    )
    res = client.get_stock_bars(req)
    df = res.df
    if df.empty:
        return df

    # Normalize multi-index and convert tz
    df = df.reset_index()  # ['symbol'(if multi), 'timestamp', ...] OR ['timestamp', ...]
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(EASTERN)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols].sort_values("timestamp")


def get_latest_quote(symbol: str) -> dict | None:
    try:
        q = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        qd = q[symbol]  # dict-like keyed by symbol
        return {
            "bid": qd.bid_price,
            "ask": qd.ask_price,
            "bid_size": qd.bid_size,
            "ask_size": qd.ask_size,
            "ts": datetime.now(EASTERN),
        }
    except Exception as e:
        st.debug(f"Quote error: {e}")
        return None


def build_candles(df: pd.DataFrame, symbol: str, quote: dict | None) -> go.Figure:
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        )
    )

    # Optional “mid” last marker from bid/ask
    if quote and quote.get("bid") and quote.get("ask"):
        last_mid = (quote["bid"] + quote["ask"]) / 2
        fig.add_trace(
            go.Scatter(
                x=[df["timestamp"].iloc[-1]],
                y=[last_mid],
                mode="markers",
                marker=dict(size=8),
                name="last (mid)",
            )
        )

    # Dynamic y padding so it never looks flat
    y_min = float(df["low"].min())
    y_max = float(df["high"].max())
    rng = y_max - y_min if y_max > y_min else y_min * 0.002
    pad = max(rng * 0.002, 0.01)

    fig.update_layout(
        title=f"{symbol} — Intraday (from market open)",
        template="plotly_white",
        height=620,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(
            title="Time (ET)",
            type="date",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title="Price ($)",
            range=[y_min - pad, y_max + pad],
            fixedrange=False,
        ),
        showlegend=True,
    )
    return fig


# ====================== SIDEBAR ======================
st.sidebar.header("Controls")

symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
tf_choice = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], index=0)
TF_MAP = {"1m": TimeFrame.Minute, "5m": TimeFrame.FiveMinutes, "15m": TimeFrame.FifteenMinutes}
timeframe = TF_MAP[tf_choice]

live_mode = st.sidebar.toggle("Live mode (auto‑refresh)", value=True)
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5)

st.sidebar.caption("Zoom/pan with mouse. Use the range slider to scrub price ranges. Times in US/Eastern.")


# ====================== AUTO-REFRESH ======================
if live_mode:
    # Re-run this script every N seconds (safe for Streamlit)
    st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")


# ====================== MAIN LAYOUT ======================
left, right = st.columns([3, 1], gap="large")

with left:
    bars = fetch_intraday_bars(symbol, timeframe)
    quote = get_latest_quote(symbol)

    if bars.empty:
        st.warning("No intraday bars returned. Market may be closed or your data plan/feed may be limited.")
    else:
        fig = build_candles(bars, symbol, quote)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Latest Quote")
    if quote:
        st.metric("Bid", f"${quote['bid']:.2f}")
        st.metric("Ask", f"${quote['ask']:.2f}")
        st.write(f"Bid size: {quote['bid_size']}")
        st.write(f"Ask size: {quote['ask_size']}")
        st.write(f"Time: {quote['ts'].strftime('%I:%M:%S %p %Z')}")
    else:
        st.info("Quote unavailable.")

    st.markdown("---")
    st.subheader("Session")
    open_et, close_et = today_open_close_et()
    now_et = datetime.now(EASTERN)
    st.write(f"Open:  {open_et.strftime('%I:%M %p %Z')}")
    st.write(f"Now:   {now_et.strftime('%I:%M:%S %p %Z')}")
    st.write(f"Close: {close_et.strftime('%I:%M %p %Z')}")

with st.expander("Show intraday data table"):
    if not bars.empty:
        st.dataframe(bars.tail(500), use_container_width=True)
