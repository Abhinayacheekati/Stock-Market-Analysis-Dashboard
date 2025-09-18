# 📈 Streamlit Stock Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

A simple, job-ready stock-analysis app built with **Streamlit**, **yfinance**, **pandas**, **plotly**, and **pandas_ta**.  
Fetch prices, visualize candles with SMAs, show RSI/MACD, and run a basic SMA crossover backtest—right in the browser.

> 🔗 **Live demo:** <YOUR_STREAMLIT_CLOUD_URL>  
> 🖼️ **Preview:**  
> <img src="assets/screenshot.png" width="800"/>

---

## 🚀 Features
- Ticker input with date/interval controls (auto-handles intraday limits)
- Candlestick chart + SMAs + volume
- RSI & MACD panels
- Simple long-only SMA crossover backtest with equity curve and metrics
- CSV export of prices + indicators
- Robust handling for empty/NaN data from yfinance

---

## 🛠️ Tech Stack
- Python, Streamlit
- yfinance, pandas, numpy
- plotly, pandas_ta

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/streamlit-stocks.git
cd streamlit-stocks

# (Windows)
py -3.11 -m venv .venv && .venv\Scripts\Activate.ps1

# (macOS/Linux)
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
