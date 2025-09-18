import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from datetime import date, timedelta

# ---------- Page config ----------
st.set_page_config(page_title="Stock Analysis (Streamlit)", page_icon="📈", layout="wide")

# ---------- Sidebar inputs ----------
st.sidebar.title("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

today = date.today()
one_year_ago = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start date", value=one_year_ago, max_value=today - timedelta(days=1))
end_date = st.sidebar.date_input("End date", value=today, min_value=start_date)

interval = st.sidebar.selectbox(
    "Interval", ["1d", "1h", "30m", "15m", "5m"], index=0,
    help="Intraday (≤60 days). Daily (1d) supports long ranges."
)

# 🔒 Auto-limit intraday to 60 days so Yahoo returns data
if interval in {"1h", "30m", "15m", "5m"}:
    max_intraday_days = 60
    if (end_date - start_date).days > max_intraday_days:
        start_date = end_date - timedelta(days=max_intraday_days)
        st.sidebar.warning(f"Intraday limited to {max_intraday_days} days. Start adjusted to {start_date}.")

use_adj = st.sidebar.checkbox("Use adjusted close (if available)", value=True)
st.sidebar.caption("Source: Yahoo Finance via yfinance")

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Strategy Params (SMA Crossover)")
short_win = st.sidebar.number_input("Short SMA", min_value=3, max_value=200, value=20, step=1)
long_win = st.sidebar.number_input("Long SMA", min_value=10, max_value=500, value=50, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Indicator Params")
rsi_len = st.sidebar.number_input("RSI length", min_value=2, max_value=200, value=14, step=1)
macd_fast = st.sidebar.number_input("MACD fast", min_value=2, max_value=50, value=12, step=1)
macd_slow = st.sidebar.number_input("MACD slow", min_value=5, max_value=100, value=26, step=1)
macd_sig = st.sidebar.number_input("MACD signal", min_value=2, max_value=50, value=9, step=1)


# ---------- Helpers ----------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if str(c) != ""]).strip() for col in df.columns.values]
    df = df.rename(columns=lambda x: str(x).strip().title())
    return df


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where all core OHLCV are NaN (yfinance sometimes returns these)
    core = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    keep_mask = df[core].notna().any(axis=1)
    return df[keep_mask]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    # First attempt: yf.download
    df = yf.download(
        ticker, start=start, end=end + timedelta(days=1),
        interval=interval, auto_adjust=False, progress=False
    )

    if df is None:
        df = pd.DataFrame()

    if not df.empty:
        df = _flatten_cols(df)
        df.index = pd.to_datetime(df.index)
        df = _ensure_cols(df)
        df = _drop_all_nan_rows(df)

    # If still empty OR Close/Adj Close all NaN, try fallback with Ticker().history
    needs_fallback = (
        df.empty or
        (("Close" in df.columns and df["Close"].dropna().empty) and
         ("Adj Close" in df.columns and df["Adj Close"].dropna().empty))
    )
    if needs_fallback:
        try:
            hist = yf.Ticker(ticker).history(
                start=start, end=end + timedelta(days=1),
                interval=interval, auto_adjust=False
            )
            if hist is not None and not hist.empty:
                hist = _flatten_cols(hist)
                hist.index = pd.to_datetime(hist.index)
                hist = _ensure_cols(hist)
                hist = _drop_all_nan_rows(hist)
                df = hist
        except Exception:
            pass

    # Final normalization
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_cols(df)
    df.index = pd.to_datetime(df.index)
    df = _ensure_cols(df)
    df = _drop_all_nan_rows(df)
    return df


def pick_close(df: pd.DataFrame, use_adj: bool) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float, name="Close_eff")

    if isinstance(df.columns, pd.MultiIndex):
        df = _flatten_cols(df)

    # Prefer Adj Close if requested and not all NaN
    if use_adj and "Adj Close" in df.columns:
        s = df["Adj Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if s.notna().any():
            return s.rename("Close_eff")

    # Fallback to Close
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.rename("Close_eff")

    return pd.Series(dtype=float, name="Close_eff")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close_eff" not in out.columns:
        return out

    out["SMA_short"] = out["Close_eff"].rolling(window=short_win, min_periods=short_win).mean()
    out["SMA_long"]  = out["Close_eff"].rolling(window=long_win,  min_periods=long_win).mean()
    out["RSI"] = ta.rsi(close=out["Close_eff"], length=rsi_len)

    macd_df = ta.macd(close=out["Close_eff"], fast=macd_fast, slow=macd_slow, signal=macd_sig)
    if macd_df is not None and not macd_df.empty:
        out["MACD"]        = macd_df.iloc[:, 0]
        out["MACD_hist"]   = macd_df.iloc[:, 1]
        out["MACD_signal"] = macd_df.iloc[:, 2]
    else:
        out["MACD"] = out["MACD_hist"] = out["MACD_signal"] = np.nan
    return out


def compute_backtest(df: pd.DataFrame) -> pd.DataFrame:
    bt = df.copy()
    if "Close_eff" not in bt.columns:
        return bt

    bt["SMA_short_eff"] = bt["Close_eff"].rolling(short_win, min_periods=short_win).mean()
    bt["SMA_long_eff"]  = bt["Close_eff"].rolling(long_win,  min_periods=long_win).mean()
    bt["position"] = np.where(bt["SMA_short_eff"] > bt["SMA_long_eff"], 1.0, 0.0)

    bt["ret"] = bt["Close_eff"].pct_change().fillna(0.0)
    bt["strategy_ret"] = bt["position"].shift(1).fillna(0.0) * bt["ret"]
    bt["equity"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["buy_hold"] = (1.0 + bt["ret"]).cumprod()
    return bt


def nice_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


# ---------- Main ----------
with st.spinner("Fetching data..."):
    prices = fetch_prices(ticker, start_date, end_date, interval)

st.title("📈 Stock Analysis Dashboard")
st.caption("Python • Streamlit • yfinance • plotly • pandas_ta")

# Debug panel (open if you need to inspect)
with st.expander("🛠 Debug (data summary)"):
    st.write("Ticker:", ticker)
    st.write("Date range:", start_date, "→", end_date, "| Interval:", interval)
    st.write("Data shape:", prices.shape)
    st.write("Columns:", list(prices.columns))
    if not prices.empty:
        st.dataframe(prices.head(10))
    st.write("Versions:", {"yfinance": yf.__version__, "pandas": pd.__version__})

if prices.empty:
    st.error("No data returned. Try a well-known ticker (e.g., AAPL, MSFT). "
             "For NSE India use suffix .NS (e.g., RELIANCE.NS). "
             "For intraday, keep the range ≤ 60 days.")
    st.stop()

# Build effective close
prices["Close_eff"] = pick_close(prices, use_adj)

if prices["Close_eff"].dropna().empty:
    st.error("No valid closing prices available for this selection.\n\n"
             "Tips:\n"
             "• Try a different ticker (AAPL, MSFT)\n"
             "• If you need NSE, use .NS suffix (e.g., TCS.NS, RELIANCE.NS)\n"
             "• For intraday, keep the range within the last 60 days\n"
             "• Try switching OFF 'Use adjusted close'")
    st.stop()

# Indicators
data = add_indicators(prices)

# ---------- KPIs ----------
colA, colB, colC, colD = st.columns(4)
cl = prices["Close_eff"].dropna()
last_close = float(cl.iloc[-1])
prev_close = float(cl.iloc[-2]) if len(cl) > 1 else last_close
chg = (last_close / prev_close - 1.0) if prev_close else 0.0

colA.metric("Ticker", ticker)
colB.metric("Last Price", f"{last_close:,.2f}")
colC.metric("Change", nice_pct(chg))
colD.metric("Period", f"{start_date} → {end_date}")

st.markdown("---")

# ---------- Price chart ----------
st.subheader("Price (Candles + SMAs) & Volume")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)

fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close_eff"], name="Price"
), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data["SMA_short"], name=f"SMA {short_win}", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_long"],  name=f"SMA {long_win}",  mode="lines"), row=1, col=1)
fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1)

fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

# ---------- Indicators ----------
with st.expander("📐 Technical Indicators (RSI & MACD)"):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**RSI**")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"))
        rsi_fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2)
        rsi_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(rsi_fig, use_container_width=True)

    with c2:
        st.markdown("**MACD**")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD"))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], mode="lines", name="Signal"))
        macd_fig.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="Hist"))
        st.plotly_chart(macd_fig, use_container_width=True)

# ---------- Backtest ----------
st.subheader("🔁 Simple SMA Crossover Backtest (Long-Only)")
bt = compute_backtest(data)

if not bt.empty and "equity" in bt.columns:
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="Strategy"))
    eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold"], name="Buy & Hold"))
    eq_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h")
    st.plotly_chart(eq_fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    start_idx, end_idx = bt.index.min(), bt.index.max()
    years = max((end_idx - start_idx).days / 365.25, 1e-9)
    equity_end = float(bt["equity"].iloc[-1])
    bh_end     = float(bt["buy_hold"].iloc[-1])
    cagr       = equity_end ** (1 / years) - 1
    cagr_bh    = bh_end ** (1 / years) - 1
    strat_ret  = bt["strategy_ret"].dropna()
    sharpe     = (strat_ret.mean() / strat_ret.std(ddof=0)) * np.sqrt(252) if not strat_ret.empty else np.nan
    roll_max   = bt["equity"].cummax()
    drawdown   = (bt["equity"] / roll_max) - 1.0
    max_dd     = float(drawdown.min())

    m1.metric("CAGR (Strategy)", nice_pct(cagr))
    m2.metric("CAGR (Buy & Hold)", nice_pct(cagr_bh))
    m3.metric("Sharpe (≈)", f"{sharpe:.2f}" if pd.notna(sharpe) else "—")
    m4.metric("Max Drawdown", nice_pct(max_dd))
else:
    st.info("Not enough data to compute backtest for this range/interval.")

# ---------- Download ----------
st.markdown("### ⬇️ Download data")
csv = data.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV (with indicators)", csv, file_name=f"{ticker}_{start_date}_{end_date}.csv", mime="text/csv")

st.caption("Educational demo only — not financial advice.")
