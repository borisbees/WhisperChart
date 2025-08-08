import os
from datetime import datetime, date, time as dtime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="WhisperChart — Intraday (Alpaca)", layout="wide")
st.title("📈 WhisperChart — Intraday (Alpaca)")

# Prefer Streamlit secrets; fall back to env vars for local dev
API_KEY = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))

if not API_KEY or not SECRET_KEY:
    st.error("⚠️ Alpaca API keys missing. Add them to `.streamlit/secrets.toml` or export env vars.")
    st.stop()

# Cache the client so we don't re-init each refresh
@st.cache_resource(show_spinner=False)
def get_client():
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

client = get_client()
EASTERN = pytz.timezone("US/Eastern")

# Get user's local timezone
try:
    USER_TZ = pytz.timezone(st.session_state.get('timezone', 'US/Eastern'))
except:
    USER_TZ = EASTERN

# Auto-detect user timezone with JavaScript (optional enhancement)
if 'timezone' not in st.session_state:
    st.session_state.timezone = 'US/Eastern'


# ====================== HELPERS ======================
def get_market_hours():
    """Get market open/close times for today in ET and user's timezone."""
    today = date.today()
    open_et = EASTERN.localize(datetime.combine(today, dtime(9, 30)))
    close_et = EASTERN.localize(datetime.combine(today, dtime(16, 0)))
    
    # Convert to user timezone
    open_local = open_et.astimezone(USER_TZ)
    close_local = close_et.astimezone(USER_TZ)
    
    return open_et, close_et, open_local, close_local


def fetch_extended_bars(symbol: str, timeframe: TimeFrame, days_back: int = 5) -> pd.DataFrame:
    """
    Fetch bars for multiple days to ensure chart has scrollable data.
    Returns tidy DataFrame with timezone-converted timestamps.
    """
    try:
        # Get extended date range
        end_time = datetime.now(EASTERN)
        start_time = end_time - timedelta(days=days_back)
        
        # Only get data during market hours to avoid gaps
        start_time = start_time.replace(hour=9, minute=30, second=0, microsecond=0)
        
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_time,
            end=end_time,
            adjustment=None,   # raw prices
            feed="iex",        # Use IEX data instead of SIP for broader compatibility
        )
        
        res = client.get_stock_bars(req)
        df = res.df
        
        if df.empty:
            return df

        # Normalize multi-index and convert timezone
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        # Convert to user's timezone for display
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(USER_TZ)
        df["timestamp_et"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(EASTERN)
        
        cols = ["timestamp", "timestamp_et", "open", "high", "low", "close", "volume"]
        available_cols = [col for col in cols if col in df.columns]
        
        return df[available_cols].sort_values("timestamp")
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def get_latest_quote(symbol: str) -> dict | None:
    """Get latest bid/ask quote with proper error handling."""
    try:
        q = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        qd = q[symbol]
        return {
            "bid": qd.bid_price,
            "ask": qd.ask_price,
            "bid_size": qd.bid_size,
            "ask_size": qd.ask_size,
            "ts": datetime.now(USER_TZ),  # User's timezone
        }
    except Exception as e:
        print(f"Quote error: {e}")  # Log error instead of st.debug
        return None


def build_enhanced_chart(df: pd.DataFrame, symbol: str, quote: dict | None) -> go.Figure:
    """
    Build an enhanced candlestick chart with better formatting and data handling.
    """
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text=f"No data available for {symbol}.<br>Market may be closed or symbol may be invalid.",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f"{symbol} — No Data Available",
            height=620,
            template="plotly_white"
        )
        return fig

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff6b6b'
        )
    )

    # Add volume as secondary trace (optional)
    if len(df) > 0 and 'volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.3,
                marker_color='lightblue'
            )
        )

    # Add current price marker if quote available
    if quote and quote.get("bid") and quote.get("ask") and not df.empty:
        last_mid = (quote["bid"] + quote["ask"]) / 2
        last_time = df["timestamp"].iloc[-1]
        
        fig.add_trace(
            go.Scatter(
                x=[last_time],
                y=[last_mid],
                mode="markers",
                marker=dict(size=10, color='red', symbol='diamond'),
                name=f"Current (${last_mid:.2f})",
            )
        )

    # Calculate price range for better y-axis scaling
    if not df.empty:
        y_min = float(df["low"].min())
        y_max = float(df["high"].max())
        price_range = y_max - y_min if y_max > y_min else y_min * 0.02
        padding = max(price_range * 0.05, 0.01)
        y_range = [y_min - padding, y_max + padding]
    else:
        y_range = None

    # Enhanced layout with better formatting
    fig.update_layout(
        title=f"{symbol} — Multi-Day Intraday Chart ({USER_TZ.zone})",
        template="plotly_white",
        height=620,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(
            title=f"Time ({USER_TZ.zone})",
            type="date",
            rangeslider=dict(
                visible=True,
                thickness=0.1
            ),
            tickformat="%m/%d %I:%M %p",  # 12-hour format with date
            tickangle=-45
        ),
        yaxis=dict(
            title="Price ($)",
            range=y_range,
            fixedrange=False,
            side="left"
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


# ====================== SIDEBAR ======================
st.sidebar.header("Controls")

symbol = st.sidebar.text_input("Symbol", value="SPY").upper()

tf_choice = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], index=1)
TF_MAP = {
    "1m": TimeFrame.Minute, 
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute)
}
timeframe = TF_MAP[tf_choice]

days_back = st.sidebar.slider("Days of data", min_value=1, max_value=10, value=3, 
                             help="More days = more scrollable data but slower loading")

live_mode = st.sidebar.toggle("Live mode (auto‑refresh)", value=True)
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15)

# Timezone selector
timezone_options = ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific', 'UTC']
selected_tz = st.sidebar.selectbox("Display timezone", timezone_options, 
                                  index=timezone_options.index(st.session_state.timezone))

if selected_tz != st.session_state.timezone:
    st.session_state.timezone = selected_tz
    USER_TZ = pytz.timezone(selected_tz)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"📍 Displaying times in **{USER_TZ.zone}**")
st.sidebar.caption("🖱️ Use mouse to zoom/pan chart")
st.sidebar.caption("📊 Use bottom slider to scrub through historical data")


# ====================== AUTO-REFRESH ======================
if live_mode:
    st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")


# ====================== MAIN LAYOUT ======================
# Show loading spinner while fetching data
with st.spinner(f"Fetching {days_back} days of {symbol} data..."):
    bars = fetch_extended_bars(symbol, timeframe, days_back)
    quote = get_latest_quote(symbol)

# Main chart
col1, col2 = st.columns([4, 1], gap="large")

with col1:
    if bars.empty:
        st.warning(f"📅 No data available for {symbol}. Market may be closed, symbol may be invalid, or your Alpaca subscription may not include this data.")
    else:
        fig = build_enhanced_chart(bars, symbol, quote)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Latest Quote")
    if quote:
        col_bid, col_ask = st.columns(2)
        with col_bid:
            st.metric("Bid", f"${quote['bid']:.2f}", help="Best bid price")
        with col_ask:
            st.metric("Ask", f"${quote['ask']:.2f}", help="Best ask price")
        
        st.write(f"**Spread:** ${abs(quote['ask'] - quote['bid']):.3f}")
        st.write(f"**Bid Size:** {quote['bid_size']:,}")
        st.write(f"**Ask Size:** {quote['ask_size']:,}")
        st.write(f"**Updated:** {quote['ts'].strftime('%I:%M:%S %p')}")
    else:
        st.info("Quote unavailable")

    st.markdown("---")
    st.subheader("🕐 Market Hours")
    
    open_et, close_et, open_local, close_local = get_market_hours()
    now_local = datetime.now(USER_TZ)
    now_et = datetime.now(EASTERN)
    
    # Show times in user's timezone
    st.write(f"**Market Open:** {open_local.strftime('%I:%M %p %Z')}")
    st.write(f"**Current Time:** {now_local.strftime('%I:%M:%S %p %Z')}")
    st.write(f"**Market Close:** {close_local.strftime('%I:%M %p %Z')}")
    
    # Market status
    if open_et <= now_et <= close_et and now_et.weekday() < 5:
        st.success("🟢 **Market is OPEN**")
    else:
        st.error("🔴 **Market is CLOSED**")

    st.markdown("---")
    st.subheader("📊 Data Stats")
    if not bars.empty:
        st.write(f"**Data Points:** {len(bars):,}")
        st.write(f"**Date Range:** {bars['timestamp'].min().strftime('%m/%d %I:%M %p')} - {bars['timestamp'].max().strftime('%m/%d %I:%M %p')}")
        if 'volume' in bars.columns:
            st.write(f"**Avg Volume:** {bars['volume'].mean():,.0f}")

# Data table in expandable section
with st.expander(f"📈 Show {symbol} Data Table (Latest 100 rows)"):
    if not bars.empty:
        # Format timestamps for display
        display_df = bars.tail(100).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%m/%d %I:%M %p')
        
        # Reorder columns for better display
        display_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if 'volume' in display_df.columns:
            display_cols.append('volume')
        
        st.dataframe(
            display_df[display_cols].iloc[::-1],  # Reverse to show newest first
            use_container_width=True,
            height=300
        )
    else:
        st.info("No data to display")

# Footer
st.markdown("---")
st.caption("💡 **Tips:** Use the range slider below the chart to navigate through historical data. Zoom in/out with mouse wheel. Live mode auto-refreshes data.")