import json
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
import openai

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📈 Stock Analyst AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analyst AI (Groq)")
st.caption("Educational purpose only – Not financial advice")

# =========================
# GROQ CONFIG
# =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"

MODEL = "openai/gpt-oss-20b"  # ✅ confirmed working

# =========================
# DATA FUNCTIONS
# =========================
def fetch_stock(symbol: str) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=450)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # Fix Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.Series(df["Close"].values.flatten(), index=df.index)

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    df["RSI"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    return df


def ask_ai(payload: dict) -> dict:
    prompt = f"""
You are a disciplined equity research assistant.

Return ONLY valid JSON in this format:

{{
  "action": "BUY" | "HOLD" | "SELL",
  "confidence": 0-100,
  "technical_summary": "",
  "fundamental_summary": "",
  "risks": [],
  "notes": ""
}}

Rules:
- BUY if price > SMA50 & SMA200 AND RSI between 45–65 AND MACD >= 0
- SELL if price < SMA200 OR RSI < 40 OR MACD < 0
- Otherwise HOLD
- Be conservative if data is limited

DATA:
{json.dumps(payload, indent=2)}
"""

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return json.loads(response["choices"][0]["message"]["content"])

# =========================
# UI
# =========================
symbol = st.text_input("Stock Symbol", placeholder="AAPL, MSFT, TSLA")
run = st.button("Analyze", type="primary")

if run and symbol:
    symbol = symbol.upper()

    with st.spinner("Fetching stock data..."):
        df = fetch_stock(symbol)

    if df.empty:
        st.error("❌ No historical data found for this symbol")
        st.stop()

    if len(df) < 60:
        st.warning(f"⚠️ Limited data available ({len(df)} days). Results may be less accurate.")

    df = compute_indicators(df)
    last = df.iloc[-1]

    payload = {
        "symbol": symbol,
        "price": float(last["Close"]),
        "sma20": float(last["SMA20"]) if not pd.isna(last["SMA20"]) else None,
        "sma50": float(last["SMA50"]) if not pd.isna(last["SMA50"]) else None,
        "sma200": float(last["SMA200"]) if not pd.isna(last["SMA200"]) else None,
        "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
        "macd": float(last["MACD"]) if not pd.isna(last["MACD"]) else None,
        "macd_signal": float(last["MACD_SIGNAL"]) if not pd.isna(last["MACD_SIGNAL"]) else None,
        "macd_hist": float(last["MACD_HIST"]) if not pd.isna(last["MACD_HIST"]) else None,
        "as_of": datetime.utcnow().isoformat() + "Z",
    }

    st.subheader("📊 Technical Metrics")
    st.json(payload)
    st.line_chart(df["Close"])

    with st.spinner("AI analyzing..."):
        result = ask_ai(payload)

    st.success(f"📌 Recommendation: **{result['action']}**")
    st.metric("Confidence", f"{result['confidence']}%")

    st.markdown("### 🧠 Technical Summary")
    st.write(result["technical_summary"])

    st.markdown("### 📘 Fundamental Summary")
    st.write(result["fundamental_summary"])

    if result.get("risks"):
        st.markdown("### ⚠️ Risks")
        for r in result["risks"]:
            st.write("•", r)

    if result.get("notes"):
        st.markdown("### 📝 Notes")
        st.write(result["notes"])

st.divider()
st.caption("Yahoo Finance · Groq · openai/gpt-oss-20b · Streamlit")
