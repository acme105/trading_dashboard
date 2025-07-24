"""
Customisable Streamlit application for monitoring NSE stock signals.

This module defines a Streamlit dashboard that periodically scans a user‑defined
set of NSE stocks using a simple high/low oscillation rule to generate buy or
sell signals.  Users can upload a CSV of instruments, select specific
symbols, choose a historical window (today's open, yesterday's open or a
configurable number of minutes before now) and adjust the refresh interval.
A progress bar conveys scanning progress and a countdown timer shows when the
next refresh will occur.

The app uses the Angel One SmartAPI via the `SmartConnect` class.  API
credentials must be supplied in a local `config.py` file with variables
`apikey`, `username`, `pwd` and `token` defined.  A `StockList.json` file is
expected in the current directory to provide default instrument metadata if
no CSV is uploaded.
"""

import time
from datetime import datetime, timedelta, time as dt_time
import json
import pandas as pd
import streamlit as st
import pyotp

from SmartApi import SmartConnect
from config import apikey, username, pwd, token


def create_connection() -> SmartConnect:
    """Initialise and return an authenticated SmartConnect instance.

    The function generates a new session on every call using a time‑based one‑
    time password (TOTP).  If the credentials are invalid, the SmartAPI
    library will raise an exception.

    Returns
    -------
    SmartConnect
        A logged‑in SmartConnect client.
    """
    obj = SmartConnect(api_key=apikey)
    # Generate a fresh session on every run.  SmartAPI requires a TOTP for
    # multi‑factor authentication.
    _ = obj.generateSession(username, pwd, pyotp.TOTP(token).now())
    return obj



def fetch_historical_data(
    obj: SmartConnect,
    exchange: str,
    stoken: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str,
) -> pd.DataFrame:
    """Retrieve historical candle data for a single instrument.

    Parameters
    ----------
    obj : SmartConnect
        Authenticated API client.
    exchange : str
        Exchange segment (e.g. "NSE").
    stoken : str
        Symbol token for the instrument.
    start_dt : datetime
        Inclusive start of the time range.
    end_dt : datetime
        Inclusive end of the time range.
    interval : str
        Candle interval (e.g. "ONE_MINUTE").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with columns Open, High, Low, Close and
        Volume.  Returns an empty DataFrame if no data is available or an
        exception occurs.
    """
    try:
        historic_param = {
            "exchange": exchange,
            "symboltoken": stoken,
            "interval": interval,
            "fromdate": start_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": end_dt.strftime("%Y-%m-%d %H:%M"),
        }
        api_response = obj.getCandleData(historic_param)
        # API returns list of lists: [datetime, open, high, low, close, volume]
        data = api_response.get("data", []) or []
        columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
        df = pd.DataFrame(data, columns=columns)
        if not df.empty:
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df.set_index("DateTime", inplace=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def compute_monitor_table(
    obj: SmartConnect,
    stock_meta: pd.DataFrame,
    from_dt: datetime,
    to_dt: datetime,
    interval: str,
    progress_bar: "st.progress" = None,
    status_text: "st.empty" = None,
) -> pd.DataFrame:
    """Compute the monitoring table for the given instruments.

    This function iterates through the provided stock metadata, fetches
    historical candles for each symbol and applies a basic oscillation rule
    to generate signals.  Optional Streamlit widgets may be supplied to
    display a progress bar and status messages.

    Parameters
    ----------
    obj : SmartConnect
        Authenticated API client.
    stock_meta : pd.DataFrame
        DataFrame with columns `symbol` and `token` describing instruments.
    from_dt : datetime
        Start of the historical window.
    to_dt : datetime
        End of the historical window.
    interval : str
        Candle interval string supported by SmartAPI (e.g. "ONE_MINUTE").
    progress_bar : st.progress, optional
        Streamlit progress bar updated as each stock is processed.
    status_text : st.empty, optional
        Streamlit placeholder updated with textual progress (e.g. "15/69 stocks checked").

    Returns
    -------
    pd.DataFrame
        DataFrame of signals sorted by score.  Columns include stock name,
        pivot values, computed ratio and strategy.
    """
    monitor_stocks = []
    total = len(stock_meta)
    for idx, row in stock_meta.iterrows():
        # Update progress widgets if provided
        if progress_bar is not None:
            progress_bar.progress((idx + 1) / total if total else 1.0)
        if status_text is not None:
            status_text.text(f"Processing {idx + 1}/{total} stocks...")

        stock_name = row["symbol"]
        stock_token = row["token"]
        data = fetch_historical_data(obj, "NSE", stock_token, from_dt, to_dt, interval)
        if data.empty:
            continue
        df = data.reset_index()
        # BUY FIRST LOGIC
        idx_h1 = df["High"].idxmax()
        h1_time = df.loc[idx_h1, "DateTime"]
        h1_value = df.loc[idx_h1, "High"]
        df_post_h1 = df[df["DateTime"] > h1_time]
        if not df_post_h1.empty:
            l1_value = df_post_h1["Low"].min()
            l1_time = df_post_h1[df_post_h1["Low"] == l1_value]["DateTime"].iloc[0]
            df_post_l1 = df[df["DateTime"] > l1_time]
            h2_value = l1_value if df_post_l1.empty else df_post_l1["High"].max()
            cp = df.iloc[-1]["Close"]
            if h1_value != l1_value:
                x = (cp - l1_value) / (h1_value - l1_value)
                y = (h2_value - l1_value) / (h1_value - l1_value)
                if (0.1 < x < 0.4) and (y < 0.6):
                    score = 2 * (h1_value - l1_value) / (h1_value + l1_value)
                    monitor_stocks.append(
                        {
                            "stock": stock_name,
                            "H1": h1_value,
                            "L1": l1_value,
                            "Next": h2_value,
                            "CP": cp,
                            "x": round(x, 3),
                            "score": round(score, 5),
                            "Strategy": "Buy first",
                        }
                    )
        # SELL FIRST LOGIC
        idx_l1 = df["Low"].idxmin()
        l1_time = df.loc[idx_l1, "DateTime"]
        l1_value = df.loc[idx_l1, "Low"]
        df_post_l1 = df[df["DateTime"] > l1_time]
        if not df_post_l1.empty:
            h1_value = df_post_l1["High"].max()
            h1_time = df_post_l1[df_post_l1["High"] == h1_value]["DateTime"].iloc[0]
            df_post_h1 = df[df["DateTime"] > h1_time]
            l2_value = h1_value if df_post_h1.empty else df_post_h1["Low"].min()
            cp = df.iloc[-1]["Close"]
            if h1_value != l1_value:
                x = (h1_value - cp) / (h1_value - l1_value)
                y = (h1_value - l2_value) / (h1_value - l1_value)
                if (0.1 < x < 0.4) and (y < 0.6):
                    score = 2 * (h1_value - l1_value) / (h1_value + l1_value)
                    monitor_stocks.append(
                        {
                            "stock": stock_name,
                            "H1": h1_value,
                            "L1": l1_value,
                            "Next": l2_value,
                            "CP": cp,
                            "x": round(x, 3),
                            "score": round(score, 5),
                            "Strategy": "Sell first",
                        }
                    )
    # Final update for progress widgets
    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_text is not None:
        status_text.text("Processing complete.")
    monitor_df = pd.DataFrame(monitor_stocks)
    if not monitor_df.empty:
        monitor_df = monitor_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return monitor_df


def load_stock_metadata() -> pd.DataFrame:
    """Load default stock metadata from StockList.json.

    The JSON file is expected to contain a list of dictionaries with at least
    `symbol`, `token`, `exch_seg` and `lotsize`.  Only NSE instruments are
    returned.
    """
    with open("StockList.json", "r") as f:
        all_stocks = json.load(f)
    df_meta = pd.DataFrame(all_stocks)
    names = pd.read_csv('shortlisted_stocks.csv')
    # Keep only NSE instruments and required columns
    df_filtered = df_meta[(df_meta["exch_seg"] == "NSE") & (df_meta["symbol"].isin(list(names["symbol"])))]
    print(f'list of stocks: {df_filtered.shape[0]}')
    return df_filtered[["symbol", "token", "lotsize"]].reset_index(drop=True)


def run_dashboard(default_refresh_seconds: int = 300) -> None:
    """Launch the interactive Streamlit dashboard.

    Parameters
    ----------
    default_refresh_seconds : int, optional
        Default refresh interval in seconds.  Users can override via the UI.
    """
    # Configure page and display title/instructions
    st.set_page_config(page_title="NSE Stock Signal Monitor", layout="wide")
    st.title("NSE Stock Signal Monitor")
    st.write(
        """
        This dashboard scans selected NSE stocks on a recurring schedule and applies
        a simple high/low oscillation rule to identify potential buy‑first and
        sell‑first setups.  Adjust the stock list, start time and refresh
        interval via the sidebar controls.  A progress bar indicates scanning
        progress, and the timer below tells you when the next refresh will
        occur.
        """
    )
    # Sidebar configuration
    st.sidebar.header("Settings")
    # File upload for custom stock list
    uploaded_file = st.sidebar.file_uploader(
        "Upload stock list (CSV)", type=["csv"],
        help="CSV should contain at least 'symbol' and 'token' columns."
    )
    # Load stock metadata
    if uploaded_file is not None:
        try:
            stock_meta_all = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            stock_meta_all = load_stock_metadata()
    else:
        stock_meta_all = load_stock_metadata()
    # Validate columns
    required_cols = {"symbol", "token"}
    if not required_cols.issubset(stock_meta_all.columns):
        st.sidebar.error("Stock list must include 'symbol' and 'token' columns.")
        return
    # Multi‑select for stock symbols
    available_symbols = stock_meta_all["symbol"].tolist()
    selected_symbols = st.sidebar.multiselect(
        "Stocks to monitor", available_symbols, default=available_symbols
    )
    stock_meta = stock_meta_all[stock_meta_all["symbol"].isin(selected_symbols)].reset_index(drop=True)
    # Start time options
    start_option = st.sidebar.selectbox(
        "Start time",
        ["Today 9:15", "Yesterday 9:15", "Minutes before now"],
        index=1,
        help="Choose the beginning of the historical window."
    )
    now_dt = datetime.now()
    # yest_date = now_dt.date() - timedelta(days=1)
    # start_dt = datetime.combine(yest_date, dt_time(9, 15))
    if start_option == "Today 9:15":
        start_dt = datetime.combine(now_dt.date(), dt_time(9, 15))
    elif start_option == "Yesterday 9:15":
        yest_date = now_dt.date() - timedelta(days=1)
        start_dt = datetime.combine(yest_date, dt_time(9, 15))
    else:
        minutes_before = int(
            st.sidebar.number_input(
                "Minutes before now", min_value=1, max_value=1440, value=15, step=1,
                help="How many minutes before the current time should the historical window start?"
            )
        )
        start_dt = now_dt - timedelta(minutes=minutes_before)
    end_dt = now_dt
    print(start_dt, end_dt)
    # Strategy dropdown (placeholder)
    _ = st.sidebar.selectbox(
        "Strategy", ["High/Low Oscillation"], index=0,
        help="Currently only one strategy is available."
    )
    # Refresh rate slider
    refresh_seconds = int(
        st.sidebar.slider(
            "Refresh interval (seconds)", 30, 900, default_refresh_seconds, step=30,
            help="Frequency at which the data is refreshed."
        )
    )
    # Placeholders for progress and data display
    timer_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    table_placeholder = st.empty()
    # Persistent state across reruns
    if "last_update" not in st.session_state:
        st.session_state["last_update"] = 0.0
    if "monitor_df" not in st.session_state:
        st.session_state["monitor_df"] = pd.DataFrame()
    if "conn" not in st.session_state:
        try:
            st.session_state["conn"] = create_connection()
        except Exception as e:
            st.error(f"Unable to login to SmartAPI: {e}")
            return
    # Determine whether refresh is required
    now_ts = time.time()
    last_update = st.session_state.get("last_update", 0.0)
    time_since_update = now_ts - last_update
    

    refreshing = False
    if time_since_update >= refresh_seconds:
        refreshing = True
        status_text.info("Refreshing data...")
        # Show previous table (don't clear it)
        df_monitor = st.session_state.get("monitor_df", pd.DataFrame())
        if not df_monitor.empty:
            table_placeholder.dataframe(df_monitor, use_container_width=True)
        # Now refresh in a spinner (UI stays visible)
        with st.spinner("Refreshing signals..."):
            conn = st.session_state["conn"]
            try:
                df_new = compute_monitor_table(
                    conn,
                    stock_meta,
                    start_dt,
                    end_dt,
                    "ONE_MINUTE",
                    progress_bar=progress_bar,
                    status_text=status_text,
                )
                st.session_state["monitor_df"] = df_new
                last_update = time.time()
                st.session_state["last_update"] = last_update
            except Exception as exc:
                status_text.error(f"Error during calculation: {exc}")

    # Always display the most recent available table (even during refresh)
    df_monitor = st.session_state.get("monitor_df", pd.DataFrame())
    if df_monitor.empty:
        table_placeholder.warning("No signals at this time.")
    else:
        table_placeholder.dataframe(df_monitor, use_container_width=True)



    # Countdown timer to next refresh
    if last_update > 0:
        next_refresh_ts = last_update + refresh_seconds
        time_left = int(max(0, next_refresh_ts - time.time()))
        next_refresh_time = datetime.fromtimestamp(next_refresh_ts).strftime("%Y-%m-%d %H:%M:%S")
        timer_placeholder.info(
            f"Next refresh in {time_left}s at {next_refresh_time}"
        )
    else:
        timer_placeholder.info("Waiting for first update...")
    # Auto rerun every second to update the countdown timer
    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    # Start the dashboard when executed as a script.  Any unhandled exceptions
    # are surfaced within the Streamlit interface.
    try:
        run_dashboard()
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")