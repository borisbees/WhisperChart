import os
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# Removed plotly import - focusing only on lightweight charts
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from streamlit.components.v1 import html as st_html
from streamlit_autorefresh import st_autorefresh
from streamlit_lightweight_charts import renderLightweightCharts

try:
    from app.proxy import set_timezone, start_proxy_server  # type: ignore
except Exception:
    start_proxy_server = None

    def set_timezone(_tz):  # type: ignore
        return None


# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="WhisperChart — Intraday Trading",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/whisperchart",
        "Report a bug": "https://github.com/your-repo/whisperchart/issues",
        "About": "WhisperChart - Real-time trading charts with Alpaca & Yahoo Finance",
    },
)

# Add custom CSS for better responsiveness
st.markdown(
    """
<style>
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Make metrics stack on mobile */
        .metric-row {
            flex-direction: column !important;
        }
        
        /* Better sidebar on mobile */
        .css-1d391kg {
            padding-top: 1rem;
        }
    }
    
    /* Better spacing for cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Improved readability */
    .big-font {
        font-size: 1.2rem !important;
        font-weight: 600;
    }
    
    /* Better code blocks */
    code {
        background-color: #f1f3f4 !important;
        padding: 2px 4px !important;
        border-radius: 4px !important;
        font-family: 'Monaco', 'Menlo', monospace !important;
    }
    
    /* Chart container responsiveness */
    .chart-container {
        width: 100%;
        height: auto;
        min-height: 400px;
    }
    
    /* Gradient headers */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title with gradient styling
st.markdown(
    '<h1 class="gradient-header">📈 WhisperChart — Intraday Trading</h1>', unsafe_allow_html=True
)

# Prefer Streamlit secrets; fall back to env vars for local dev
API_KEY = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))

if not API_KEY or not SECRET_KEY:
    st.error("⚠️ Alpaca API keys missing. Add them to `.streamlit/secrets.toml` or export env vars.")
    st.stop()


# Cache the client so we don't re-init each refresh
@st.cache_resource(show_spinner=False)
def get_client():
    # Create client for paper trading (free accounts)
    # Note: Market data access depends on your Alpaca subscription
    return StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)


# Enhanced caching for data with time-based TTL
@st.cache_data(ttl=300, show_spinner=False)  # 5 minute cache
def get_cached_data(symbol: str, timeframe_str: str, days_back: int, cache_key: str):
    """
    Cached data fetching with intelligent cache key.
    Cache key includes symbol, timeframe, days, and timestamp bucket for cache invalidation.
    """
    return None  # Placeholder - actual implementation in fetch functions


@st.cache_data(ttl=60, show_spinner=False)  # 1 minute cache for quotes
def get_cached_quote(symbol: str, cache_key: str):
    """Cached quote fetching with 1-minute TTL."""
    return None  # Placeholder - actual implementation in quote function


client = get_client()
EASTERN = pytz.timezone("US/Eastern")

# Auto-detect user's local timezone

try:
    # Get system timezone
    local_tz_name = time.tzname[0] if not time.daylight else time.tzname[1]
    # Try to map to pytz timezone
    if hasattr(time, "tzname") and time.tzname:
        # Common timezone mappings
        tz_mapping = {
            "EST": "US/Eastern",
            "EDT": "US/Eastern",
            "CST": "US/Central",
            "CDT": "US/Central",
            "MST": "US/Mountain",
            "MDT": "US/Mountain",
            "PST": "US/Pacific",
            "PDT": "US/Pacific",
            "UTC": "UTC",
            "GMT": "UTC",
        }
        detected_tz = tz_mapping.get(local_tz_name, "US/Eastern")
    else:
        detected_tz = "US/Eastern"

    USER_TZ = pytz.timezone(detected_tz)
except Exception:
    # Fallback to Eastern if detection fails
    USER_TZ = EASTERN


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


def fetch_yahoo_data(symbol: str, timeframe: TimeFrame, days_back: int = 5) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance as fallback when Alpaca is unavailable.
    """
    try:
        # Map timeframe to Yahoo Finance intervals
        if timeframe == TimeFrame.Minute:
            interval = "1m"
            period_days = min(days_back, 7)  # Yahoo limits 1m data to 7 days
        elif timeframe == TimeFrame(5, TimeFrameUnit.Minute):
            interval = "5m"
            period_days = min(days_back, 60)  # 5m data available for 60 days
        elif timeframe == TimeFrame(15, TimeFrameUnit.Minute):
            interval = "15m"
            period_days = min(days_back, 60)
        else:
            interval = "5m"  # Default
            period_days = days_back

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            prepost=False,
        )

        if df.empty:
            return df

        # Rename columns to match Alpaca format
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Reset index to get timestamp as column
        df = df.reset_index()
        df = df.rename(columns={"Datetime": "timestamp"})

        # Convert timezone to user's timezone
        if df["timestamp"].dt.tz is None:
            # Yahoo data comes in market timezone (ET), localize it first
            df["timestamp"] = df["timestamp"].dt.tz_localize(EASTERN)

        df["timestamp"] = df["timestamp"].dt.tz_convert(USER_TZ)
        df["timestamp_et"] = df["timestamp"].dt.tz_convert(EASTERN)

        # Select only the columns we need
        cols = ["timestamp", "timestamp_et", "open", "high", "low", "close", "volume"]
        available_cols = [col for col in cols if col in df.columns]

        return df[available_cols].sort_values("timestamp")

    except Exception as e:
        st.warning(f"Yahoo Finance fallback also failed for {symbol}: {e}")
        return pd.DataFrame()


def create_cache_key(symbol: str, timeframe: TimeFrame, days_back: int) -> str:
    """Create intelligent cache key with time bucketing for optimal cache hits."""
    # Create time bucket (every 5 minutes for intraday, hourly for daily)
    now = datetime.now()
    if timeframe == TimeFrame.Minute:
        bucket = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    else:
        bucket = now.replace(minute=0, second=0, microsecond=0)

    return f"{symbol}_{timeframe}_{days_back}_{bucket.isoformat()}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_chunk(
    symbol: str,
    timeframe: TimeFrame,
    start_time: datetime,
    end_time: datetime,
    source: str = "alpaca",
) -> pd.DataFrame:
    """
    Fetch a specific chunk of data with caching.
    This allows for efficient loading of only needed data ranges.
    """
    try:
        if source == "yahoo":
            return fetch_yahoo_data_chunk(symbol, timeframe, start_time, end_time)
        else:
            return fetch_alpaca_data_chunk(symbol, timeframe, start_time, end_time)
    except Exception as e:
        st.warning(f"Failed to fetch {source} data chunk: {e}")
        return pd.DataFrame()


def fetch_alpaca_data_chunk(
    symbol: str, timeframe: TimeFrame, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """Fetch specific time range from Alpaca with optimized parameters."""
    try:
        client = get_client()
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_time,
            end=end_time,
            adjustment=None,
        )

        res = client.get_stock_bars(req)
        df = res.df

        if df.empty:
            return df

        # Normalize and convert timezone
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(USER_TZ)

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [col for col in cols if col in df.columns]

        return df[available_cols].sort_values("timestamp")

    except Exception as e:
        raise e


def fetch_yahoo_data_chunk(
    symbol: str, timeframe: TimeFrame, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """Fetch specific time range from Yahoo Finance."""
    try:
        # Map timeframe
        if timeframe == TimeFrame.Minute:
            interval = "1m"
        elif timeframe == TimeFrame(5, TimeFrameUnit.Minute):
            interval = "5m"
        elif timeframe == TimeFrame(15, TimeFrameUnit.Minute):
            interval = "15m"
        else:
            interval = "5m"

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_time.strftime("%Y-%m-%d"),
            end=end_time.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            prepost=False,
        )

        if df.empty:
            return df

        # Format for consistency
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df = df.reset_index()
        df = df.rename(columns={"Datetime": "timestamp"})

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(EASTERN)

        df["timestamp"] = df["timestamp"].dt.tz_convert(USER_TZ)

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [col for col in cols if col in df.columns]

        return df[available_cols].sort_values("timestamp")

    except Exception as e:
        raise e


def fetch_extended_bars(symbol: str, timeframe: TimeFrame, days_back: int = 5) -> pd.DataFrame:
    """
    Optimized data fetching with chunked loading and intelligent caching.
    """
    # Create cache key for this request
    cache_key = create_cache_key(symbol, timeframe, days_back)

    # Check if we can use cached data
    if f"data_cache_{cache_key}" in st.session_state:
        cached_data = st.session_state[f"data_cache_{cache_key}"]
        cache_time = st.session_state.get(f"cache_time_{cache_key}", datetime.min)

        # Use cache if less than 5 minutes old (or 1 minute for 1m timeframe)
        cache_ttl = 60 if timeframe == TimeFrame.Minute else 300
        if (datetime.now() - cache_time).total_seconds() < cache_ttl:
            return cached_data

    try:
        # Calculate time ranges for chunked loading
        end_time = datetime.now(EASTERN)
        start_time = end_time - timedelta(days=days_back)

        # For performance, limit initial load and enable progressive loading
        if days_back > 5:
            # Load recent data first for immediate display
            recent_start = end_time - timedelta(days=2)
            recent_data = fetch_data_chunk(symbol, timeframe, recent_start, end_time, "alpaca")

            if recent_data.empty:
                # Fallback to Yahoo
                recent_data = fetch_data_chunk(symbol, timeframe, recent_start, end_time, "yahoo")

            # Store recent data immediately for fast display
            if not recent_data.empty:
                st.session_state[f"data_cache_{cache_key}"] = recent_data
                st.session_state[f"cache_time_{cache_key}"] = datetime.now()

                # Schedule background loading of older data
                if f"loading_full_data_{symbol}" not in st.session_state:
                    st.session_state[f"loading_full_data_{symbol}"] = True
                    # This would be handled by a background task in a full implementation

                return recent_data

        # Standard loading for reasonable date ranges
        full_data = fetch_data_chunk(symbol, timeframe, start_time, end_time, "alpaca")

        if full_data.empty:
            # Try Yahoo Finance fallback
            st.info(f"📊 Alpaca data unavailable for {symbol}, trying Yahoo Finance fallback...")
            full_data = fetch_yahoo_data(symbol, timeframe, days_back)

            if not full_data.empty:
                st.success(f"✅ Successfully loaded {symbol} data from Yahoo Finance!")

        # Cache the result
        if not full_data.empty:
            st.session_state[f"data_cache_{cache_key}"] = full_data
            st.session_state[f"cache_time_{cache_key}"] = datetime.now()

        return full_data

    except Exception as e:
        # Enhanced error handling with fallback options
        return handle_data_fetch_error(symbol, timeframe, days_back, str(e))


def handle_data_fetch_error(
    symbol: str, timeframe: TimeFrame, days_back: int, error_msg: str
) -> pd.DataFrame:
    """Handle data fetch errors with intelligent fallbacks."""
    now_et = datetime.now(EASTERN)
    is_weekend = now_et.weekday() >= 5

    # Try Yahoo Finance as final fallback
    try:
        yahoo_data = fetch_yahoo_data(symbol, timeframe, days_back)
        if not yahoo_data.empty:
            st.success(f"✅ Successfully loaded {symbol} data from Yahoo Finance!")
            return yahoo_data
    except Exception:
        pass

    # Show appropriate error message
    if "403" in error_msg or "Forbidden" in error_msg:
        if is_weekend:
            st.warning(
                f"""
            **📅 Weekend Data Limitation for {symbol}**
            
            Data feeds are limited on weekends. Try:
            - **Different symbols**: SPY, AAPL, TSLA, MSFT
            - **Larger intervals**: 5m or 15m instead of 1m
            - **Fewer days**: Reduce historical data range
            """
            )
        else:
            st.error(
                f"""
            **🔐 Market Data Access Error for {symbol}**
            
            Enable market data access in your [Alpaca dashboard](https://app.alpaca.markets/)
            or try different symbols like AAPL, TSLA, MSFT.
            """
            )
    else:
        st.error(f"Error fetching data for {symbol}: {error_msg}")

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


def build_lightweight_chart_config(df: pd.DataFrame, symbol: str, quote: dict | None) -> dict:
    """
    Build configuration for lightweight charts with candlestick data.
    """
    if df.empty:
        return None

    # Convert DataFrame to lightweight charts format
    candlestick_data = []
    volume_data = []

    for _, row in df.iterrows():
        # Use date string format instead of timestamp for better compatibility
        time_str = row["timestamp"].strftime("%Y-%m-%d %I:%M:%S %p")
        candlestick_data.append(
            {
                "time": time_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )

        if "volume" in df.columns and pd.notna(row["volume"]):
            volume_data.append(
                {
                    "time": time_str,
                    "value": float(row["volume"]),
                    "color": "#26a69a" if row["close"] >= row["open"] else "#ef5350",
                }
            )

    # Chart configuration with responsive height
    chart_config = {
        "layout": {
            "background": {"type": "solid", "color": "#ffffff"},
            "textColor": "#333333",
        },
        "grid": {
            "vertLines": {"color": "#e0e0e0"},
            "horzLines": {"color": "#e0e0e0"},
        },
        "timeScale": {
            "timeVisible": True,
            "secondsVisible": False,
            "borderColor": "#cccccc",
        },
        "rightPriceScale": {
            "borderColor": "#cccccc",
        },
        "crosshair": {
            "mode": 0,  # Normal crosshair mode
        },
        "width": 0,  # Auto width
        "height": 400,  # Fixed height for consistency
    }

    # Series data
    series_data = [
        {
            "type": "candlestick",
            "data": candlestick_data,
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderUpColor": "#26a69a",
            "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
        }
    ]

    # Add volume series if available
    if volume_data and len(volume_data) > 0:
        series_data.append(
            {
                "type": "histogram",
                "data": volume_data,
                "priceFormat": {
                    "type": "volume",
                },
                "priceScaleId": "volume",
                "scaleMargins": {
                    "top": 0.8,
                    "bottom": 0,
                },
            }
        )

    return {"chart": chart_config, "series": series_data}


# ====================== ENHANCED SIDEBAR ======================
st.sidebar.markdown(
    """
<div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           border-radius: 8px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">⚡ Controls</h2>
</div>
""",
    unsafe_allow_html=True,
)

# Symbol input with better styling
st.sidebar.markdown("**📊 Symbol**")
symbol = st.sidebar.text_input(
    "Enter stock symbol",
    value="SPY",
    placeholder="e.g., AAPL, TSLA, MSFT",
    help="Enter any valid stock ticker symbol",
).upper()

# Time interval with description
st.sidebar.markdown("**⏱️ Time Interval**")
tf_choice = st.sidebar.selectbox(
    "Chart granularity",
    ["1m", "5m", "15m"],
    index=1,
    help="1m = 1 minute bars, 5m = 5 minute bars, etc.",
)
TF_MAP = {
    "1m": TimeFrame.Minute,
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
}
timeframe = TF_MAP[tf_choice]

# Progressive prefetch plan (no UI): start small, extend quietly
if timeframe == TimeFrame.Minute:
    initial_days, max_days, step_days = 1, 5, 2
elif timeframe == TimeFrame(5, TimeFrameUnit.Minute):
    initial_days, max_days, step_days = 2, 10, 3
elif timeframe == TimeFrame(15, TimeFrameUnit.Minute):
    initial_days, max_days, step_days = 3, 14, 3
else:
    initial_days, max_days, step_days = 3, 14, 3

prefetch_key = f"prefetch_progress_{symbol}_{tf_choice}"
target_key = f"prefetch_target_{symbol}_{tf_choice}"
if target_key not in st.session_state:
    st.session_state[target_key] = max_days
days_back = int(st.session_state.get(prefetch_key, initial_days))

# Intelligent caching always enabled
enable_caching = True

# AI Prediction Features (MVP)
st.sidebar.markdown("---")
st.sidebar.markdown("**🔮 AI Predictions**")
show_prediction = st.sidebar.checkbox(
    "🎯 Prediction Line",
    value=False,
    help="Show AI-powered price forecast extending to session close",
)

show_confidence = st.sidebar.checkbox(
    "📊 Confidence Ribbon",
    value=False,
    help="Show uncertainty bands around predictions (advanced feature)",
)

if show_prediction or show_confidence:
    horizon_minutes = st.sidebar.slider(
        "Forecast horizon (minutes)",
        min_value=15,
        max_value=240,
        value=60,
        help="How far into the future to predict",
    )
else:
    horizon_minutes = 60

# Live mode section
st.sidebar.markdown("---")
st.sidebar.markdown("**🔄 Live Updates**")
live_mode = st.sidebar.checkbox(
    "Auto-refresh", value=False, help="Automatically refresh data at specified intervals"
)

if live_mode:
    refresh_secs = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=15,
        help="How often to fetch new data",
    )
else:
    refresh_secs = 15  # Default value when live mode is off

# Auto-detected timezone info (no manual selection)
st.sidebar.markdown("---")
st.sidebar.markdown("**🌍 Timezone Info**")
st.sidebar.info(f"📍 Auto-detected: **{USER_TZ.zone}**")
st.sidebar.caption("All times shown in your local timezone")

# Help section
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tips & Info**")
st.sidebar.info(f"📍 Times shown in **{USER_TZ.zone}**")
st.sidebar.success("🖱️ Mouse: zoom/pan chart")
st.sidebar.success("📊 Slider: scrub historical data")

# Weekend notice
now_et = datetime.now(EASTERN)
if now_et.weekday() >= 5:  # Weekend
    st.sidebar.warning("📅 Weekend: Using Yahoo Finance data")


# ====================== AUTO-REFRESH ======================
if live_mode:
    st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")


# ====================== OPTIMIZED MAIN LAYOUT ======================
# Performance-optimized data loading with progress indicators

# Simplified, faster data loading
start_time = time.time()

# Simple cache check (no complex progressive loading for now)
cache_key = f"{symbol}_{timeframe}_{days_back}"
cache_timeout = 60 if timeframe == TimeFrame.Minute else 300  # 1-5 min cache

if enable_caching and cache_key in st.session_state:
    cached_data, cache_timestamp = st.session_state[cache_key]
    if (time.time() - cache_timestamp) < cache_timeout:
        bars = cached_data
        quote = get_latest_quote(symbol)
        load_time = time.time() - start_time
        st.info(f"⚡ Loaded from cache in {load_time:.2f}s")
    else:
        # Cache expired, fetch new data
        bars = fetch_extended_bars(symbol, timeframe, days_back)
        quote = get_latest_quote(symbol)
        if enable_caching and not bars.empty:
            st.session_state[cache_key] = (bars, time.time())
        load_time = time.time() - start_time
        st.info(f"📊 Fresh data loaded in {load_time:.2f}s")
else:
    # No cache, fetch new data
    with st.spinner(f"Loading {symbol} data..."):
        bars = fetch_extended_bars(symbol, timeframe, days_back)
        quote = get_latest_quote(symbol)
        if enable_caching and not bars.empty:
            st.session_state[cache_key] = (bars, time.time())
    load_time = time.time() - start_time
    st.info(f"📊 Data loaded in {load_time:.2f}s")

# After initial render, progressively prefetch more history up to target quietly
pref_target = int(st.session_state.get(target_key, max_days))
if days_back < pref_target:
    # Bump progress and store for next run
    st.session_state[prefetch_key] = min(days_back + step_days, pref_target)
    # Trigger a one-shot, very short autorefresh to fetch the next chunk
    if not live_mode:
        st_autorefresh(interval=400, limit=1, key=f"prefetch_{symbol}_{tf_choice}")

# Add data loading performance stats
if not bars.empty:
    load_time = st.session_state.get(f"load_time_{cache_key}", 0)
    data_points = len(bars)
    st.caption(f"📊 Loaded {data_points:,} data points | ⚡ {load_time:.1f}s")

# ====================== RESPONSIVE MAIN LAYOUT ======================
# Use responsive columns that adapt to screen size
col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    # Chart header with better formatting and timezone info
    current_time = datetime.now(USER_TZ)
    st.markdown(
        f"""
    <div style="padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #333;">📈 {symbol} — Multi-Day Intraday Chart</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
            🌍 Timezone: {USER_TZ.zone} | 🕐 Current: {current_time.strftime('%b %d, %I:%M %p')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if bars.empty:
        st.error(
            f"""
        **📅 No data available for {symbol}**
        
        This could be because:
        - Market is closed and data feeds are limited
        - Symbol doesn't exist or is invalid
        - Try different symbols: SPY, AAPL, TSLA, MSFT
        """
        )
    else:
        # Build chart configuration
        chart_config = build_lightweight_chart_config(bars, symbol, quote)

        if chart_config is None:
            st.error("Failed to build chart configuration")
            st.info("📊 Showing raw data instead:")
            st.dataframe(bars.head(10), use_container_width=True)
        else:
            # Enhanced debug info
            with st.expander("🔍 Debug Info (Chart Data)"):
                st.write("**Chart Config Structure:**")
                st.json(
                    {
                        "chart_keys": list(chart_config.keys()),
                        "series_count": len(chart_config.get("series", [])),
                        "data_points": (
                            len(chart_config["series"][0]["data"])
                            if chart_config.get("series")
                            else 0
                        ),
                        "sample_data": (
                            chart_config["series"][0]["data"][:3]
                            if chart_config.get("series") and chart_config["series"][0].get("data")
                            else []
                        ),
                    }
                )

                st.write("**Live Data Debug:**")
                if not bars.empty:
                    st.write(f"Bars shape: {bars.shape}")
                    st.write(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
                    st.write("**First 3 candlestick data points:**")
                    _candles = locals().get("candlestick_data")
                    if _candles and len(_candles) > 0:
                        for i, point in enumerate(_candles[:3]):
                            timestamp_readable = pd.to_datetime(point["time"], unit="s")
                            st.write(
                                f"Point {i+1}: {timestamp_readable} - O:{point['open']} H:{point['high']} L:{point['low']} C:{point['close']}"
                            )

            # Render via a custom HTML component with fetch-on-scroll; fall back if unavailable
            try:
                import json

                # Start or reuse proxy server (absolute import via app package)
                if "_proxy_port" not in st.session_state:
                    try:
                        set_timezone(USER_TZ)  # type: ignore
                    except Exception:
                        pass
                    st.session_state._proxy_port = (
                        start_proxy_server() if start_proxy_server else None
                    )
                proxy_port = st.session_state.get("_proxy_port")

                if not proxy_port:
                    raise RuntimeError("Proxy server unavailable")

                chart_data = [
                    {
                        "time": int(r["timestamp"].timestamp()),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                    }
                    for _, r in bars.iterrows()
                ]
                initial_json = json.dumps(chart_data)
                symbol_js = json.dumps(symbol)
                tf_js = json.dumps(tf_choice)
                base_url_js = json.dumps(f"http://127.0.0.1:{proxy_port}")

                html = f"""
<div id=\"wc_container\" style=\"position: relative; height: 500px;\">
  <div id=\"wc_chart\" style=\"height: 100%;\"></div>
  <div id=\"wc_loading\" style=\"position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; padding: 4px 8px; border-radius: 6px; font: 12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif; display: none;\">Loading more…</div>
</div>
<script src=\"https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js\"></script>
<script>
  const container = document.getElementById('wc_chart');
  const loadingEl = document.getElementById('wc_loading');
  const chart = LightweightCharts.createChart(container, {{
    layout: {{ background: {{ type: 'solid', color: 'white' }}, textColor: '#222' }},
    timeScale: {{ 
      timeVisible: true, 
      secondsVisible: false, 
      borderColor: '#D1D4DC',
      rightOffset: 5,
      barSpacing: 3
    }},
    rightPriceScale: {{ 
      borderColor: '#D1D4DC',
      scaleMargins: {{ top: 0.1, bottom: 0.1 }}
    }},
    grid: {{ 
      vertLines: {{ color: 'rgba(197,203,206,0.5)' }}, 
      horzLines: {{ color: 'rgba(197,203,206,0.5)' }} 
    }},
    localization: {{
      locale: 'en-US',
      timeFormatter: (timestamp) => {{
        const date = new Date(timestamp * 1000);
        const hours = date.getHours();
        const minutes = date.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        const displayHours = hours % 12 || 12;
        return `${{displayHours}}:${{minutes.toString().padStart(2, '0')}} ${{ampm}}`;
      }},
      dateFormatter: (timestamp) => {{
        const date = new Date(timestamp * 1000);
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const month = months[date.getMonth()];
        const day = date.getDate();
        return `${{month}} ${{day}}`;
      }}
    }}
  }});
  const series = chart.addCandlestickSeries({{
    upColor: '#26a69a', downColor: '#ef5350', borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350'
  }});

  let data = {initial_json};
  series.setData(data);

  let earliest = data.length ? data[0].time : Math.floor(Date.now()/1000);
  const baseUrl = {base_url_js};
  const symbol = {symbol_js};
  const timeframe = {tf_js};
  let loading = false;

  async function fetchMore(beforeTs) {{
    try {{
      if (loadingEl) loadingEl.style.display = 'block';
      const resp = await fetch(`${{baseUrl}}/history?symbol=${{encodeURIComponent(symbol)}}&timeframe=${{encodeURIComponent(timeframe)}}&before=${{beforeTs}}&limit=800`);
      if (!resp.ok) return;
      const payload = await resp.json();
      const more = payload.data || [];
      if (more.length) {{
        data = more.concat(data);
        series.setData(data);
        earliest = data[0].time;
      }}
    }} catch (e) {{ console.log('fetchMore error', e); }}
    finally {{ if (loadingEl) loadingEl.style.display = 'none'; }}
  }}

  chart.timeScale().subscribeVisibleTimeRangeChange((range) => {{
    if (!range || loading || !earliest) return;
    const margin = 60 * 20; // 20 minutes in seconds
    if (range.from && (range.from - earliest) < margin) {{
      loading = true;
      fetchMore(earliest).finally(() => loading = false);
    }}
  }});
</script>
                """

                st_html(html, height=520, scrolling=False)

                if not bars.empty:
                    start_time = bars["timestamp"].min().strftime("%b %d, %Y %I:%M %p")
                    end_time = bars["timestamp"].max().strftime("%b %d, %Y %I:%M %p")
                    st.caption(
                        f"📅 Time range: {start_time} → {end_time} ({USER_TZ.zone}) | 📊 {len(chart_data)} bars"
                    )

                with st.expander("🔍 Chart Debug Info", expanded=False):
                    st.write("**Sample Chart Data (first 3 points):**")
                    st.json(chart_data[:3] if len(chart_data) >= 3 else chart_data)
                    st.write("**Data Summary:**")
                    st.write(f"- Total bars: {len(chart_data)}")
                    st.write(
                        f"- Time range: {min(d['time'] for d in chart_data)} to {max(d['time'] for d in chart_data)}"
                    )
                    st.write(
                        f"- Price range: ${min(d['low'] for d in chart_data):.2f} - ${max(d['high'] for d in chart_data):.2f}"
                    )

            except Exception as e:
                st.error(f"Chart rendering failed (custom): {e}")
                st.exception(e)
                st.info("📊 Falling back to built-in renderer")
                try:
                    chart_data = [
                        {
                            "time": int(r["timestamp"].timestamp()),
                            "open": float(r["open"]),
                            "high": float(r["high"]),
                            "low": float(r["low"]),
                            "close": float(r["close"]),
                        }
                        for _, r in bars.iterrows()
                    ]
                    chart_config = {
                        "chart": {
                            "layout": {
                                "textColor": "black",
                                "background": {"type": "solid", "color": "white"},
                            },
                            "timeScale": {
                                "timeVisible": True,
                                "borderColor": "#D1D4DC",
                                "rightOffset": 5,
                            },
                            "localization": {"locale": "en-US"},
                        },
                        "series": [{"type": "Candlestick", "data": chart_data}],
                    }
                    renderLightweightCharts(
                        [chart_config], key=f"chart_fallback_{symbol}_{timeframe}_{len(chart_data)}"
                    )
                except Exception as e2:
                    st.error(f"Fallback renderer failed: {e2}")
                    
                    # Try alternative time format as last resort
                    st.info("🔄 Trying alternative time format...")
                    try:
                        # Try with date string format for daily aggregation
                        alt_candlestick_data = []

                        # Group by date and aggregate OHLC
                        daily_bars = bars.copy()
                        # Simple date grouping
                        daily_bars["date"] = daily_bars["timestamp"].dt.date

                        daily_agg = (
                            daily_bars.groupby("date")
                            .agg(
                                {
                                    "open": "first",
                                    "high": "max",
                                    "low": "min",
                                    "close": "last",
                                    "volume": "sum" if "volume" in daily_bars.columns else "count",
                                }
                            )
                            .reset_index()
                        )

                        # Convert to chart format
                        for _, row in daily_agg.iterrows():
                            alt_candlestick_data.append(
                                {
                                    "time": row["date"].strftime("%Y-%m-%d"),
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                }
                            )

                        # Try rendering with daily aggregated data
                        series_data = [
                            {
                                "type": "Candlestick",
                                "data": alt_candlestick_data,
                                "options": {
                                    "upColor": "#26a69a",
                                    "downColor": "#ef5350",
                                    "borderUpColor": "#26a69a",
                                    "borderDownColor": "#ef5350",
                                    "wickUpColor": "#26a69a",
                                    "wickDownColor": "#ef5350",
                                },
                            }
                        ]

                        try:
                            chart_options = {
                                "layout": {
                                    "textColor": "black",
                                    "background": {"type": "solid", "color": "white"},
                                },
                                "timeScale": {
                                    "timeVisible": True,
                                    "borderColor": "#D1D4DC",
                                    "rightOffset": 5,
                                },
                                "localization": {"locale": "en-US"},
                            }

                            renderLightweightCharts(
                                [{"chart": chart_options, "series": series_data}],
                                key=f"chart_alt_{symbol}_{timeframe}_{len(alt_candlestick_data)}",
                            )
                        except Exception as e3:
                            st.error(f"Alternative renderer failed: {e3}")
                            st.info("📊 Showing raw data instead")
                            st.dataframe(bars.head(20), use_container_width=True)
                    except Exception as e4:
                        st.error(f"Alternative time format failed: {e4}")
                        st.info("📊 Showing raw data instead")
                        st.dataframe(bars.head(20), use_container_width=True)


        # Responsive metrics grid
        if not bars.empty:
            st.markdown("---")

            # Mobile-friendly metrics layout
            latest_price = bars["close"].iloc[-1]
            price_change = latest_price - bars["open"].iloc[0]
            price_change_pct = (price_change / bars["open"].iloc[0]) * 100

            # Use responsive columns for metrics
            metric_cols = st.columns([1, 1, 1, 1])

            with metric_cols[0]:
                st.metric(
                    label="💰 Latest Price",
                    value=f"${latest_price:.2f}",
                    help="Most recent closing price",
                )

            with metric_cols[1]:
                st.metric(
                    label="📈 Change",
                    value=f"${price_change:.2f}",
                    delta=f"{price_change_pct:.2f}%",
                    help="Price change from period start",
                )

            with metric_cols[2]:
                if "volume" in bars.columns and not pd.isna(bars["volume"].iloc[-1]):
                    latest_vol = bars["volume"].iloc[-1]
                    st.metric(
                        label="📊 Volume",
                        value=f"{latest_vol:,.0f}",
                        help="Most recent trading volume",
                    )
                else:
                    st.metric(label="📊 Volume", value="N/A")

            with metric_cols[3]:
                data_source = (
                    "📊 Yahoo Finance" if "yahoo" in str(type(bars)).lower() else "🦙 Alpaca"
                )
                st.metric(label="🔗 Data Source", value=data_source, help="Source of market data")

with col2:
    # ====================== RIGHT SIDEBAR - RESPONSIVE ======================

    # Quote Section
    st.markdown(
        """
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="margin: 0 0 1rem 0; color: #333;">💰 Latest Quote</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if quote:
        # Responsive quote layout
        quote_cols = st.columns(2)
        with quote_cols[0]:
            st.metric("Bid", f"${quote['bid']:.2f}", help="Best bid price")
        with quote_cols[1]:
            st.metric("Ask", f"${quote['ask']:.2f}", help="Best ask price")

        # Quote details with better formatting
        spread = abs(quote["ask"] - quote["bid"])
        st.markdown(
            f"""
        **Spread:** `${spread:.3f}`  
        **Bid Size:** `{quote['bid_size']:,}`  
        **Ask Size:** `{quote['ask_size']:,}`  
        **Updated:** `{quote['ts'].strftime('%I:%M:%S %p')}`
        """
        )
    else:
        st.info("📵 Quote unavailable (market closed)")

    # Market Hours Section
    st.markdown("---")
    st.markdown(
        """
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="margin: 0 0 1rem 0; color: #333;">🕐 Market Hours</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    open_et, close_et, open_local, close_local = get_market_hours()
    now_local = datetime.now(USER_TZ)
    now_et = datetime.now(EASTERN)

    # Market status with better visual indicator
    if open_et <= now_et <= close_et and now_et.weekday() < 5:
        st.success("🟢 **Market is OPEN**")
        status_color = "#28a745"
    else:
        st.error("🔴 **Market is CLOSED**")
        status_color = "#dc3545"

    # Times with better formatting
    st.markdown(
        f"""
    **Market Open:** `{open_local.strftime('%I:%M %p %Z')}`  
    **Current Time:** `{now_local.strftime('%I:%M:%S %p %Z')}`  
    **Market Close:** `{close_local.strftime('%I:%M %p %Z')}`  
    """
    )

    # Data Stats Section
    st.markdown("---")
    st.markdown(
        """
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="margin: 0 0 1rem 0; color: #333;">📊 Data Stats</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not bars.empty:
        data_points = len(bars)
        # Display times in user's timezone with proper formatting
        start_time = bars["timestamp"].min()
        end_time = bars["timestamp"].max()

        # Convert to user timezone for display
        try:
            if hasattr(start_time, "tz_convert"):
                start_time = start_time.tz_convert(USER_TZ)
                end_time = end_time.tz_convert(USER_TZ)
        except Exception:
            pass  # Use as-is if conversion fails

        date_start = start_time.strftime("%b %d, %I:%M %p")
        date_end = end_time.strftime("%b %d, %I:%M %p")

        st.markdown(
            f"""
        **Data Points:** `{data_points:,}`  
        **Date Range:**  
        `{date_start}` to `{date_end}`  
        **Timezone:** `{USER_TZ.zone}`
        """
        )

        if "volume" in bars.columns and not bars["volume"].isna().all():
            avg_volume = bars["volume"].mean()
            st.markdown(f"**Avg Volume:** `{avg_volume:,.0f}`")
    else:
        st.info("No data loaded yet")

# Data table in expandable section
with st.expander(f"📈 Show {symbol} Data Table (Latest 100 rows)"):
    if not bars.empty:
        # Format timestamps for display
        display_df = bars.tail(100).copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%m/%d %I:%M %p")

        # Reorder columns for better display
        display_cols = ["timestamp", "open", "high", "low", "close"]
        if "volume" in display_df.columns:
            display_cols.append("volume")

        st.dataframe(
            display_df[display_cols].iloc[::-1],  # Reverse to show newest first
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No data to display")

# Footer
st.markdown("---")
st.caption(
    "💡 **Tips:** Zoom and pan the chart to explore data. Enable prediction features to see AI forecasts. Live mode auto-refreshes data for real-time trading."
)
