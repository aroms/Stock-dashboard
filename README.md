# Stock Analysis Dashboard

A plug-and-play Streamlit app so you can **look at a dashboard instead of studying** and still make informed, rules-based decisions.

## Features
- Ticker chart with SMA overlays, RSI, and MACD
- Heuristic **Signals** panel (RSI zones, MACD & SMA crossovers)
- **Backtest** tab for SMA crossover strategy with CAGR, max drawdown, and Sharpe
- **Portfolio** simulator for up to 8 tickers
- Caching for snappy reloads

## Quick Start
1. Ensure you have Python 3.10+ installed.
2. Create & activate a virtual environment (recommended).
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Data is fetched from Yahoo Finance via `yfinance`. Availability can vary by ticker/interval.
- This tool is for **education**, not financial advice. Use your own judgment and risk controls.