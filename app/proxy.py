import threading
import socket
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pytz

# Default timezones (will be overridden by main app)
EASTERN = pytz.timezone('US/Eastern')
USER_TZ = pytz.timezone('US/Pacific')  # Default, will be overridden

def set_timezone(user_tz):
    """Set the timezone for the proxy server (called by main app)"""
    global USER_TZ
    USER_TZ = user_tz


def _pick_free_port(preferred_start: int = 8750, preferred_end: int = 8799) -> int:
    for port in range(preferred_start, preferred_end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Let OS pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _map_timeframe(tf: str) -> TimeFrame:
    tf = tf.strip().lower()
    if tf in {"1m", "1min", "1"}:
        return TimeFrame.Minute
    if tf in {"5m", "5min", "5"}:
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf in {"15m", "15min", "15"}:
        return TimeFrame(15, TimeFrameUnit.Minute)
    # default
    return TimeFrame(5, TimeFrameUnit.Minute)

def fetch_data_chunk(symbol: str, timeframe: TimeFrame, start_dt: datetime, end_dt: datetime, source: str = "alpaca") -> pd.DataFrame:
    """Fetch a chunk of historical data for the proxy server (lazy imports)."""
    if source == "alpaca":
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest

            client = StockHistoricalDataClient()
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
            )
            bars = client.get_stock_bars(request)
            if bars and hasattr(bars, "df") and not bars.df.empty:
                return bars.df
        except Exception:
            pass

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        if timeframe == TimeFrame.Minute:
            interval = "1m"
        elif hasattr(timeframe, "amount"):
            interval = f"{timeframe.amount}m"
        else:
            interval = "5m"

        df = ticker.history(start=start_dt, end=end_dt, interval=interval)
        if not df.empty:
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df.index.name = "timestamp"
            df = df.reset_index()
            return df
    except Exception:
        pass

    return pd.DataFrame()


def start_proxy_server() -> Optional[int]:
    """Start a local FastAPI server to serve historical bars for the front-end component.

    Returns the port if started successfully, otherwise None.
    """
    try:
        # Lazy imports to avoid hard dependency at module import time
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn

        app = FastAPI(title="WhisperChart Proxy", version="0.1.0")

        # CORS for cross-port requests from Streamlit
        try:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://127.0.0.1:8501", "http://localhost:8501", "*"],
                allow_credentials=False,
                allow_methods=["GET"],
                allow_headers=["*"],
            )
        except Exception:
            pass

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/history")
        def history(
            symbol: str = Query(..., min_length=1, max_length=12),
            timeframe: str = Query("5m"),
            before: int = Query(..., description="Unix seconds; fetch bars strictly before this time"),
            limit: int = Query(500, ge=10, le=2000),
        ):
            try:
                tf = _map_timeframe(timeframe)
                end_dt = datetime.fromtimestamp(before, tz=USER_TZ)
                # Estimate start by minutes ~ limit * step
                step_min = 1 if tf == TimeFrame.Minute else tf.amount if hasattr(tf, 'amount') else 5
                start_dt = end_dt - timedelta(minutes=step_min * limit)

                # Fetch chunk using existing optimized function
                df: pd.DataFrame = fetch_data_chunk(symbol, tf, start_dt, end_dt, source="alpaca")
                if df.empty:
                    # Fallback to yahoo chunk
                    df = fetch_data_chunk(symbol, tf, start_dt, end_dt, source="yahoo")

                data = []
                if not df.empty:
                    for _, row in df.iterrows():
                        t = int(pd.to_datetime(row["timestamp"]).tz_convert(USER_TZ).timestamp())
                        if t < before:  # ensure strictly before 'before'
                            data.append(
                                {
                                    "time": t,
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                }
                            )

                # Sort data by timestamp (ascending)
                data = sorted(data, key=lambda d: d["time"])
                return JSONResponse({"symbol": symbol.upper(), "timeframe": timeframe, "data": data})
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=str(e))

        port = _pick_free_port()

        def _run():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return port
    except Exception:
        return None
