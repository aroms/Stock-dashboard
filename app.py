import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# ========================
# Helpers
# ========================
@st.cache_data(show_spinner=False)
def load_price_data(ticker, start, end, interval):
    """
    Try Yahoo Finance (yfinance). If empty or fails, fall back to Stooq via pandas-datareader.
    Stooq provides daily data; we resample to weekly/monthly when requested.
    """
    # --- 1) Try Yahoo via yfinance ---
    try:
        import yfinance as yf
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data, pd.DataFrame) and not data.empty:
            # If multi-index columns (multi-ticker), flatten
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [
                    "_".join([c for c in col if c]) for col in data.columns
                ]
            return data
    except Exception as e:
        st.warning(f"Yahoo fetch failed: {e}")

    # --- 2) Fallback to Stooq (no API key) via pandas-datareader ---
    try:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(ticker, data_source="stooq")  # newest->oldest
        df = df.sort_index()

        # Keep only requested date range
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

        # Stooq columns: Open, High, Low, Close, Volume (already), but ensure case
        df = df.rename(columns=str.title)

        # Resample if user requested weekly or monthly
        if interval == "1wk":
            df = df.resample("W-FRI").last()
        elif interval == "1mo":
            df = df.resample("M").last()

        # Match yfinance-style expectations (need Close at least)
        return df
    except Exception as e:
        st.error("No data from Yahoo or Stooq. Try another ticker/period, or refresh.")
        return pd.DataFrame()


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()


def cagr(equity_curve, dates):
    if len(equity_curve) < 2:
        return np.nan
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1


def sharpe(returns, rf=0.0, periods_per_year=252):
    mean = np.nanmean(returns)
    std = np.nanstd(returns)
    if std == 0 or np.isnan(std):
        return np.nan
    return (mean - rf / periods_per_year) / std * np.sqrt(periods_per_year)


def signal_summary(close, rsi_series, macd_line, macd_signal, sma_fast, sma_slow):
    signals = []
    if len(close) >= 2 and not np.isnan(rsi_series.iloc[-1]):
        if rsi_series.iloc[-1] <= 30:
            signals.append("RSI oversold (≤30): potential rebound")
        elif rsi_series.iloc[-1] >= 70:
            signals.append("RSI overbought (≥70): potential pullback")
    if (
        len(macd_line) >= 2
        and not np.isnan(macd_line.iloc[-1])
        and not np.isnan(macd_signal.iloc[-1])
    ):
        if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2]:
            signals.append("MACD bullish crossover")
        elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2]:
            signals.append("MACD bearish crossover")
    if len(sma_fast.dropna()) and len(sma_slow.dropna()):
        if sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] <= sma_slow.iloc[-2]:
            signals.append("SMA crossover: fast above slow (bullish)")
        elif sma_fast.iloc[-1] < sma_slow.iloc[-1] and sma_fast.iloc[-2] >= sma_slow.iloc[-2]:
            signals.append("SMA crossover: fast below slow (bearish)")
    if not signals:
        signals = ["No strong technical signals right now."]
    return signals


# ========================
# Sidebar Inputs
# ========================
st.sidebar.title("Settings")

default_ticker = "AAPL"
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, MSFT, SPY):", value=default_ticker).upper()

period_choice = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    index=3,
)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

today = datetime.today().date()
if period_choice == "max":
    start_date = "1900-01-01"
else:
    mapping = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 365 * 2,
        "5y": 365 * 5,
        "10y": 365 * 10,
    }
    start_date = (today - timedelta(days=mapping[period_choice])).isoformat()
end_date = today.isoformat()

fast_len = st.sidebar.number_input("Fast MA length", min_value=3, max_value=200, value=20, step=1)
slow_len = st.sidebar.number_input("Slow MA length", min_value=5, max_value=400, value=50, step=1)
rsi_len = st.sidebar.number_input("RSI length", min_value=5, max_value=50, value=14, step=1)

with st.sidebar.expander("Backtest (SMA crossover)"):
    bt_fast = st.number_input("Fast SMA", min_value=3, max_value=200, value=20, step=1, key="bt_fast")
    bt_slow = st.number_input("Slow SMA", min_value=5, max_value=400, value=50, step=1, key="bt_slow")
    initial_capital = st.number_input(
        "Initial capital ($)", min_value=1000, max_value=1_000_000, value=10_000, step=500
    )
    use_short = st.checkbox("Allow short on bearish", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the tabs to jump between Chart, Signals, Backtest, and Portfolio.")


# ========================
# Data Fetch
# ========================
data = load_price_data(ticker, start_date, end_date, interval)
if data.empty:
    st.error("No data returned. Check the ticker or try a different period/interval.")
    st.stop()

# Ensure we have Close column
if "Close" not in data.columns:
    close_candidates = [c for c in data.columns if c.lower().startswith("close")]
    if close_candidates:
        data["Close"] = data[close_candidates[0]]
    else:
        st.error("Couldn't find a Close price column in the data.")
        st.stop()

close = data["Close"].astype(float)

# Indicators
sma_fast_series = sma(close, fast_len)
sma_slow_series = sma(close, slow_len)
rsi_series = rsi(close, rsi_len)
macd_line, macd_signal, macd_hist = macd(close)

# ========================
# Header
# ========================
st.title("📈 Stock Analysis Dashboard")
st.caption("Enter a ticker in the sidebar, then explore the tabs below. Nothing here is financial advice.")

# ========================
# Tabs
# ========================
tab_chart, tab_signals, tab_backtest, tab_portfolio = st.tabs(["Chart", "Signals", "Backtest", "Portfolio"])

with tab_chart:
    st.subheader(f"Price & Indicators — {ticker}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=close.index, y=sma_fast_series, mode="lines", name=f"SMA {fast_len}"))
    fig.add_trace(go.Scatter(x=close.index, y=sma_slow_series, mode="lines", name=f"SMA {slow_len}"))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RSI**")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=close.index, y=rsi_series, mode="lines", name="RSI"))
        fig_rsi.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)
    with col2:
        st.markdown("**MACD**")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=close.index, y=macd_line, mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=close.index, y=macd_signal, mode="lines", name="Signal"))
        fig_macd.add_trace(go.Bar(x=close.index, y=macd_hist, name="Histogram"))
        fig_macd.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_macd, use_container_width=True)

with tab_signals:
    st.subheader("Actionable Signals (heuristics)")
    signals = signal_summary(close, rsi_series, macd_line, macd_signal, sma_fast_series, sma_slow_series)
    for s in signals:
        st.write("• " + s)

    st.markdown("---")
    st.caption("These are simple, rule-based heuristics. Combine with your own judgment.")

with tab_backtest:
    st.subheader(f"SMA Crossover Backtest — {ticker}")
    close_bt = close.dropna()
    sma_fast_bt = sma(close_bt, bt_fast)
    sma_slow_bt = sma(close_bt, bt_slow)

    # Trading logic: long when fast > slow, else (optional) short; fully invested, no transaction costs.
    position = np.where(sma_fast_bt > sma_slow_bt, 1, -1 if use_short else 0)
    position = pd.Series(position, index=close_bt.index).shift(1).fillna(0)  # trade on next bar
    returns = close_bt.pct_change().fillna(0)
    strat_returns = position * returns
    equity = (1 + strat_returns).cumprod()
    equity = equity * (initial_capital / equity.iloc[0]) if len(equity) > 0 else equity

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name="Strategy Equity"))
    buy_hold_curve = (1 + returns).cumprod()
    buy_hold_curve = buy_hold_curve * (initial_capital / buy_hold_curve.iloc[0])
    fig_bt.add_trace(go.Scatter(x=buy_hold_curve.index, y=buy_hold_curve, mode="lines", name="Buy & Hold"))
    fig_bt.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig_bt, use_container_width=True)

    dd = max_drawdown(equity.values) if len(equity) else np.nan
    r_annual = cagr(equity.values, equity.index) if len(equity) else np.nan
    sharpe_ratio = sharpe(strat_returns.values) if len(strat_returns) else np.nan

    colA, colB, colC = st.columns(3)
    colA.metric("CAGR", f"{(r_annual*100):.2f}%" if pd.notna(r_annual) else "—")
    colB.metric("Max Drawdown", f"{(dd*100):.2f}%" if pd.notna(dd) else "—")
    colC.metric("Sharpe (no RF)", f"{sharpe_ratio:.2f}" if pd.notna(sharpe_ratio) else "—")

    st.caption("Model assumptions: next-bar execution, no costs/slippage, continuous allocation. Past performance ≠ future results.")

with tab_portfolio:
    st.subheader("Simple Portfolio Simulator")
    st.caption("Combine up to 8 tickers and fixed weights to see historical performance.")

    default_list = "AAPL, MSFT, AMZN, NVDA"
    tickers_input = st.text_input("Tickers (comma-separated)", value=default_list)
    weights_input = st.text_input("Weights (comma-separated, must sum to 1.0)", value="0.25, 0.25, 0.25, 0.25")
    pf_interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, key="pf_interval")
    pf_period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2, key="pf_period")

    if st.button("Run Portfolio Simulation"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        try:
            weights = [float(x.strip()) for x in weights_input.split(",")]
        except Exception:
            st.error("Weights must be numbers.")
            st.stop()

        if len(tickers) != len(weights):
            st.error("The number of tickers and weights must match.")
            st.stop()
        if abs(sum(weights) - 1.0) > 1e-6:
            st.error("Weights must sum to 1.0.")
            st.stop()
        if len(tickers) > 8:
            st.error("Please limit to 8 tickers.")
            st.stop()

        # Determine start date
        today = datetime.today().date()
        if pf_period == "max":
            pf_start = "1900-01-01"
        else:
            mapping = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
            pf_start = (today - timedelta(days=mapping[pf_period])).isoformat()
        pf_end = today.isoformat()

        # Download with fallback for each ticker, then assemble a price panel
        frames = []
        for t in tickers:
            df = load_price_data(t, pf_start, pf_end, pf_interval)
            if df.empty or "Close" not in df.columns:
                continue
            frames.append(df[["Close"]].rename(columns={"Close": t}))
        if not frames:
            st.error("No data for portfolio simulation.")
            st.stop()
        panel = pd.concat(frames, axis=1).dropna()

        returns = panel.pct_change().dropna()
        weights_arr = np.array(weights[: len(panel.columns)])
        if len(weights_arr) != len(panel.columns):
            st.error("Some tickers had no data; adjust weights to match available tickers.")
            st.stop()
        pf_returns = (returns * weights_arr).sum(axis=1)
        pf_equity = (1 + pf_returns).cumprod()

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=pf_equity.index, y=pf_equity, mode="lines", name="Portfolio"))
        for col in panel.columns:
            fig_pf.add_trace(
                go.Scatter(
                    x=(1 + returns[col]).cumprod().index,
                    y=(1 + returns[col]).cumprod(),
                    mode="lines",
                    name=col,
                    opacity=0.5,
                )
            )
        fig_pf.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig_pf, use_container_width=True)

        st.write("**Stats**")
        col1, col2, col3 = st.columns(3)
        col1.metric("CAGR (portfolio)", f"{(cagr(pf_equity.values, pf_equity.index)*100):.2f}%")
        col2.metric("Max Drawdown", f"{(max_drawdown(pf_equity.values)*100):.2f}%")
        col3.metric("Sharpe (no RF)", f"{sharpe(pf_returns.values):.2f}")

        st.dataframe(panel.tail(10))

st.markdown("---")
st.caption("Built with Streamlit • Yahoo via yfinance (fallback: Stooq via pandas-datareader) • Educational use only")
