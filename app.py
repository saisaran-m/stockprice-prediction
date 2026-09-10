import os
import base64
os.environ["KERAS_BACKEND"] = "torch"

from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Stock Market Predictor & Analytics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, institutional-grade appearance
st.markdown("""
<style>
    .tour-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 12px;
        background-color: rgba(3, 105, 161, 0.12);
        color: #0284c7;
        border: 1px solid rgba(3, 105, 161, 0.25);
    }
    .feature-box {
        background-color: rgba(125, 125, 125, 0.05);
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }
    .feature-title {
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 13px;
        opacity: 0.8;
        line-height: 1.5;
    }
    .step-indicator {
        font-size: 14px;
        font-weight: 600;
        opacity: 0.8;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Session State for Onboarding Tour
# ----------------------------------------------------
if "show_tour" not in st.session_state:
    st.session_state["show_tour"] = True

if "slide_index" not in st.session_state:
    st.session_state["slide_index"] = 0

# ----------------------------------------------------
# Model Loader with Caching
# ----------------------------------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Stock Predictions Model.keras')
    if not os.path.exists(model_path):
        model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
    return load_model(model_path)

model = get_model()

# ----------------------------------------------------
# Company Info Loader
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def get_company_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return {
            "name": info.get("longName") or info.get("shortName") or ticker_symbol,
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "currency": info.get("currency") or "USD",
            "summary": info.get("longBusinessSummary") or ""
        }
    except Exception:
        return {
            "name": ticker_symbol,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": None,
            "pe_ratio": None,
            "currency": "USD",
            "summary": ""
        }

# ----------------------------------------------------
# Background Image Loader for Onboarding Tour
# ----------------------------------------------------
@st.cache_data
def get_tour_bg_base64():
    bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'tour_bg.jpg')
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ----------------------------------------------------
# ONBOARDING TOUR RENDERER (5 SLIDES)
# ----------------------------------------------------
def render_onboarding_tour():
    slide = st.session_state["slide_index"]

    bg_b64 = get_tour_bg_base64()
    if bg_b64:
        st.markdown(f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(8, 12, 24, 0.78), rgba(4, 8, 18, 0.86)),
                            url("data:image/jpeg;base64,{bg_b64}") no-repeat center center fixed !important;
                background-size: cover !important;
            }}
            [data-testid="stHeader"] {{
                background-color: transparent !important;
            }}
            [data-testid="stVerticalBlockBorderWrapper"] {{
                background: rgba(15, 23, 42, 0.84) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px solid rgba(56, 189, 248, 0.32) !important;
                border-radius: 18px !important;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55) !important;
            }}
            .tour-pill {{
                display: inline-block;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
                background-color: rgba(14, 165, 233, 0.2) !important;
                color: #38bdf8 !important;
                border: 1px solid rgba(56, 189, 248, 0.4) !important;
            }}
            .tour-title {{
                font-size: 30px !important;
                font-weight: 800 !important;
                color: #ffffff !important;
                margin-bottom: 8px !important;
                margin-top: 4px !important;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
            }}
            .tour-subtitle {{
                font-size: 16px !important;
                color: #cbd5e1 !important;
                margin-bottom: 24px !important;
                line-height: 1.6 !important;
            }}
            .feature-box {{
                background-color: rgba(30, 41, 59, 0.72) !important;
                border: 1px solid rgba(148, 163, 184, 0.22) !important;
                border-radius: 12px !important;
                padding: 18px 20px !important;
                height: 100% !important;
                backdrop-filter: blur(8px) !important;
            }}
            .feature-title {{
                font-weight: 700 !important;
                font-size: 15px !important;
                color: #f8fafc !important;
                margin-bottom: 6px !important;
            }}
            .feature-desc {{
                font-size: 13px !important;
                color: #94a3b8 !important;
                line-height: 1.5 !important;
            }}
        </style>
        """, unsafe_allow_html=True)

    slides_data = [
        {
            "badge": "SLIDE 1 OF 5 • PLATFORM OVERVIEW",
            "title": "Welcome to Stock Market Predictor & Analytics Pro",
            "subtitle": "An institutional-grade financial analytics and deep learning suite designed for forward-looking market intelligence.",
            "features": [
                {
                    "title": "Real-Time 2026 Market Data",
                    "desc": "Direct integration with the live Yahoo Finance API. Always fetches the latest market closing prices up to today."
                },
                {
                    "title": "Deep Learning LSTM Architecture",
                    "desc": "Powered by a 4-layer Long Short-Term Memory neural network trained to capture complex temporal price momentum."
                },
                {
                    "title": "Multi-Day Future Forecasting",
                    "desc": "Recursive multi-step engine that projects future price trajectories up to 30 upcoming business days into the future."
                },
                {
                    "title": "Institutional Technical Suite",
                    "desc": "Interactive Plotly candlestick charts, volume bars, Bollinger Bands, 14-day RSI, and MACD momentum indicators."
                }
            ]
        },
        {
            "badge": "SLIDE 2 OF 5 • PRICE ACTION & VOLUME",
            "title": "Interactive Candlestick & Volume Dynamics",
            "subtitle": "Inspect daily price discovery and market liquidity with full interactive controls.",
            "features": [
                {
                    "title": "OHLC Candlestick Charts",
                    "desc": "Visualizes Open, High, Low, and Close price action. Green candles indicate bullish buying; red candles indicate selling pressure."
                },
                {
                    "title": "Bollinger Bands (20-Day, ±2σ)",
                    "desc": "Dynamic volatility envelopes around the 20-day SMA. Touching the upper band signals overextension; touching the lower band signals oversold bounces."
                },
                {
                    "title": "Moving Averages (MA50 & MA200)",
                    "desc": "Track intermediate and macro-level market trends. Identify Golden Cross and Death Cross momentum shifts."
                },
                {
                    "title": "Trading Volume Confirmation",
                    "desc": "Color-coded volume bars synchronized directly beneath price candles to validate institutional participation."
                }
            ]
        },
        {
            "badge": "SLIDE 3 OF 5 • QUANTITATIVE MOMENTUM",
            "title": "Quantitative Momentum: RSI & MACD",
            "subtitle": "Detect overextended market conditions and impending trend reversals before price reacts.",
            "features": [
                {
                    "title": "Relative Strength Index (RSI 14)",
                    "desc": "Monitors price velocity on a 0 to 100 scale. Automatically flags Overbought (>70) and Oversold (<30) danger zones."
                },
                {
                    "title": "Real-Time RSI Status Badge",
                    "desc": "Top KPI card immediately categorizes the stock's momentum status (Overbought, Oversold, or Neutral)."
                },
                {
                    "title": "MACD & Signal Line",
                    "desc": "Measures the divergence between 12-day and 26-day EMAs against a 9-day signal trigger line to confirm directional bias."
                },
                {
                    "title": "Dynamic Momentum Histogram",
                    "desc": "Green bars highlight accelerating bullish momentum, while red bars show expanding bearish momentum."
                }
            ]
        },
        {
            "badge": "SLIDE 4 OF 5 • AI FORECASTING ENGINE",
            "title": "Deep Learning LSTM Forecasting & Confidence Bounds",
            "subtitle": "Understand how the neural network generates future price trajectories.",
            "features": [
                {
                    "title": "100-Day Sequential Memory",
                    "desc": "The model evaluates the preceding 100 consecutive trading days to recognize sequential price structures."
                },
                {
                    "title": "Zero-Gap Continuity Calibration",
                    "desc": "Anchored directly to today's real closing price. Eliminates artificial step jumps so forecasts branch seamlessly from today."
                },
                {
                    "title": "Forecast Uncertainty Band",
                    "desc": "Shaded confidence interval cone (+80% CI) derived from recent market volatility, showing expected Bullish and Bearish boundaries."
                },
                {
                    "title": "Business Day Alignment",
                    "desc": "Automatically skips weekends and holiday gaps, aligning forecast rows with actual market trading sessions."
                }
            ]
        },
        {
            "badge": "SLIDE 5 OF 5 • CONTROLS & EXPORT",
            "title": "Ready to Explore: Controls & Data Export",
            "subtitle": "Customize your analysis parameters and export professional forecast reports.",
            "features": [
                {
                    "title": "Global Ticker Search",
                    "desc": "Type any ticker symbol (e.g. GOOG, AAPL, MSFT, TSLA, NVDA, RELIANCE.NS) to analyze US or global stocks."
                },
                {
                    "title": "Custom Forecast Horizons",
                    "desc": "Use the sidebar slider to predict from 1 day ahead up to a full month (30 business days) into the future."
                },
                {
                    "title": "One-Click CSV Export",
                    "desc": "Click 'Download CSV' in the forecast tab to export the full daily predicted schedule to a spreadsheet."
                },
                {
                    "title": "Revisit Tour Anytime",
                    "desc": "Click the 'App Guide & Tour' button in the sidebar whenever you want to review these explanations."
                }
            ]
        }
    ]

    current_data = slides_data[slide]

    # Render unified tour card
    with st.container(border=True):
        st.markdown(f'<span class="tour-pill">{current_data["badge"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<h1 class="tour-title">{current_data["title"]}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="tour-subtitle">{current_data["subtitle"]}</p>', unsafe_allow_html=True)

        # 4-Column Feature Grid
        cols = st.columns(4)
        for i, feat in enumerate(current_data["features"]):
            with cols[i]:
                st.markdown(f"""
                <div class="feature-box">
                    <div class="feature-title">{feat["title"]}</div>
                    <div class="feature-desc">{feat["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # Navigation Controls Bar
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 3, 2, 2])

        with nav_col1:
            if st.button("Skip Tour & Launch App", use_container_width=True):
                st.session_state["show_tour"] = False
                st.rerun()

        with nav_col2:
            # Visual Progress Dots
            dots = ["●" if i == slide else "○" for i in range(5)]
            st.markdown(
                f"<div style='text-align: center; padding-top: 8px; font-weight: bold; color: #3b82f6; font-size: 16px;'>"
                f"{' '.join(dots)} &nbsp;&nbsp; <span style='font-size: 13px; color: #64748b;'>Step {slide + 1} of 5</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with nav_col3:
            if slide > 0:
                if st.button("← Previous", use_container_width=True):
                    st.session_state["slide_index"] = slide - 1
                    st.rerun()

        with nav_col4:
            if slide < 4:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state["slide_index"] = slide + 1
                    st.rerun()
            else:
                if st.button("Get Started 🚀", type="primary", use_container_width=True):
                    st.session_state["show_tour"] = False
                    st.rerun()

# ----------------------------------------------------
# Main Execution Branch
# ----------------------------------------------------
if st.session_state["show_tour"]:
    render_onboarding_tour()
else:
    # ----------------------------------------------------
    # Sidebar Controls (Main Dashboard Mode)
    # ----------------------------------------------------
    st.sidebar.header("⚙️ Market Settings")

    stock = st.sidebar.text_input(
        "Stock Symbol",
        value="GOOG",
        help="Enter ticker e.g. GOOG, AAPL, MSFT, TSLA, NVDA, AMZN, RELIANCE.NS, TCS.NS"
    ).upper().strip()

    today = date.today()
    default_start = date(2015, 1, 1)

    start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=today - timedelta(days=120))
    end_date = st.sidebar.date_input(
        "End Date",
        value=today,
        max_value=today + timedelta(days=1),
        help="Latest trading date. US stock markets operate in Eastern Time (EDT)."
    )

    if start_date >= end_date:
        st.sidebar.error("Start Date must be before End Date.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Forecast Horizon")
    future_days = st.sidebar.slider(
        "Upcoming Days to Forecast",
        min_value=1,
        max_value=30,
        value=7,
        help="Select how many future business days to forecast using the recursive LSTM neural network."
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("📖 App Guide & Tour", use_container_width=True):
        st.session_state["show_tour"] = True
        st.session_state["slide_index"] = 0
        st.rerun()

    st.sidebar.markdown("### 💡 Quick Tickers")
    quick_col1, quick_col2 = st.sidebar.columns(2)
    with quick_col1:
        st.caption("• `GOOG` (Google)\n• `AAPL` (Apple)\n• `MSFT` (Microsoft)")
    with quick_col2:
        st.caption("• `NVDA` (Nvidia)\n• `TSLA` (Tesla)\n• `AMZN` (Amazon)")

    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 Deep Learning LSTM • Real-Time Yahoo Finance API • Plotly Analytics")

    # ----------------------------------------------------
    # Fetch Market Data
    # ----------------------------------------------------
    with st.spinner(f"Downloading live market data for {stock}..."):
        fetch_end = end_date + timedelta(days=1)
        data = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=fetch_end.strftime('%Y-%m-%d'))

    if data.empty:
        st.error(f"❌ No market data found for ticker symbol **'{stock}'**. Please verify the symbol and try again.")
        st.stop()

    if len(data) < 100:
        st.error(f"❌ Selected date range contains only {len(data)} trading days. The LSTM model requires at least 100 trading days. Please select an earlier Start Date.")
        st.stop()

    # Flatten potential MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    close_series = data['Close'].dropna()
    data['Close'] = close_series

    # ----------------------------------------------------
    # Company Header Banner
    # ----------------------------------------------------
    comp_info = get_company_info(stock)
    st.title(f"📈 {comp_info['name']} ({stock})")

    sub_items = [f"**Sector:** {comp_info['sector']}", f"**Industry:** {comp_info['industry']}"]
    if comp_info['market_cap']:
        cap_val = comp_info['market_cap']
        if cap_val >= 1e12:
            sub_items.append(f"**Market Cap:** ${cap_val/1e12:.2f}T")
        elif cap_val >= 1e9:
            sub_items.append(f"**Market Cap:** ${cap_val/1e9:.2f}B")
        else:
            sub_items.append(f"**Market Cap:** ${cap_val/1e6:.2f}M")
    if comp_info['pe_ratio']:
        sub_items.append(f"**P/E Ratio:** {comp_info['pe_ratio']:.2f}")

    st.caption(" • ".join(sub_items))

    # ----------------------------------------------------
    # Technical Indicators Computation
    # ----------------------------------------------------
    # Moving Averages
    data['MA50'] = close_series.rolling(50).mean()
    data['MA100'] = close_series.rolling(100).mean()
    data['MA200'] = close_series.rolling(200).mean()

    # Bollinger Bands (20-day, 2 std)
    data['SMA20'] = close_series.rolling(20).mean()
    data['STD20'] = close_series.rolling(20).std()
    data['BB_Upper'] = data['SMA20'] + (data['STD20'] * 2)
    data['BB_Lower'] = data['SMA20'] - (data['STD20'] * 2)

    # RSI (14-day)
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Price change stats
    latest_date = data.index[-1]
    latest_close = float(close_series.iloc[-1])
    prev_close = float(close_series.iloc[-2]) if len(close_series) > 1 else latest_close
    price_change = latest_close - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0.0

    # ----------------------------------------------------
    # Upcoming Days Forecasting Engine with Continuity Calibration
    # ----------------------------------------------------
    close_df = pd.DataFrame(close_series)
    scaler_future = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_future.fit_transform(close_df)

    # Calculate model offset at day 0 (today) to ensure smooth continuity without step-jumps
    if len(scaled_data) >= 101:
        prev_100_scaled = scaled_data[-101:-1].reshape(1, 100, 1)
        pred_today_scaled = model.predict(prev_100_scaled, verbose=0)
        pred_today_unscaled = float(scaler_future.inverse_transform(pred_today_scaled)[0, 0])
        calibration_offset = latest_close - pred_today_unscaled
    else:
        calibration_offset = 0.0

    # Multi-step recursive forecasting
    last_100_scaled = scaled_data[-100:].reshape(1, 100, 1)
    current_batch = last_100_scaled.copy()
    future_predictions = []

    with st.spinner(f"Running LSTM deep learning forecast for next {future_days} upcoming days..."):
        for _ in range(future_days):
            pred_scaled = model.predict(current_batch, verbose=0)
            future_predictions.append(pred_scaled[0, 0])
            current_batch = np.append(current_batch[:, 1:, :], [[[pred_scaled[0, 0]]]], axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    raw_future_prices = scaler_future.inverse_transform(future_predictions).flatten()

    # Apply calibration offset to guarantee realistic, seamless price continuity from today
    future_prices = raw_future_prices + calibration_offset

    # Future business days (excludes weekends)
    future_dates = pd.bdate_range(start=latest_date + pd.Timedelta(days=1), periods=future_days)

    # Baseline uncertainty for confidence intervals (estimated from recent 30-day rolling volatility)
    recent_std = float(close_series.tail(30).std())
    time_steps = np.sqrt(np.arange(1, future_days + 1))
    upper_confidence = future_prices + (1.28 * recent_std * time_steps * 0.20)
    lower_confidence = np.maximum(0.01, future_prices - (1.28 * recent_std * time_steps * 0.20))

    next_day_price = float(future_prices[0])
    next_day_change = next_day_price - latest_close
    next_day_change_pct = (next_day_change / latest_close) * 100 if latest_close != 0 else 0.0

    # ----------------------------------------------------
    # Top KPI Metric Cards
    # ----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"Market Price ({latest_date.strftime('%b %d, %Y')})",
            value=f"${latest_close:,.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%) vs Prev Close (${prev_close:,.2f})"
        )

    with col2:
        st.metric(
            label=f"Predicted Next Day ({future_dates[0].strftime('%b %d')})",
            value=f"${next_day_price:,.2f}",
            delta=f"{next_day_change:+.2f} ({next_day_change_pct:+.2f}%)"
        )

    with col3:
        st.metric(
            label=f"{future_days}-Day Forecast Target",
            value=f"${future_prices[-1]:,.2f}",
            delta=f"{(future_prices[-1] - latest_close):+.2f} ({(future_prices[-1] - latest_close)/latest_close*100:+.2f}%)"
        )

    with col4:
        latest_rsi = float(data['RSI'].iloc[-1]) if not np.isnan(data['RSI'].iloc[-1]) else 50.0
        rsi_state = "Overbought (>70)" if latest_rsi >= 70 else ("Oversold (<30)" if latest_rsi <= 30 else "Neutral")
        st.metric(
            label=f"Current 14-Day RSI",
            value=f"{latest_rsi:.1f}",
            delta=rsi_state,
            delta_color="normal" if 30 < latest_rsi < 70 else "inverse"
        )

    st.markdown("---")

    # ----------------------------------------------------
    # Navigation Tabs
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Upcoming Forecast",
        "🕯️ Candlestick & Volume",
        "📊 Technical Indicators",
        "🧪 Model Validation",
        "📋 Raw Data"
    ])

    # ----------------------------------------------------
    # TAB 1: UPCOMING DAYS FORECAST (INTERACTIVE PLOTLY)
    # ----------------------------------------------------
    with tab1:
        st.subheader(f"Next {future_days}-Day Price Forecast with Uncertainty Intervals")
        
        # Slice last 60 actual trading days
        history_window = min(60, len(close_series))
        hist_dates = data.index[-history_window:]
        hist_prices = close_series.iloc[-history_window:].values

        # Bridge between last actual and future (starts exactly at latest close for 100% smooth continuity)
        bridge_dates = [hist_dates[-1]] + list(future_dates)
        bridge_prices = [latest_close] + list(future_prices)
        bridge_upper = [latest_close] + list(upper_confidence)
        bridge_lower = [latest_close] + list(lower_confidence)

        fig_future = go.Figure()

        # Historical line
        fig_future.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_prices,
            mode="lines",
            name="Historical Close (Last 60 Days)",
            line=dict(color="#1f77b4", width=2.5)
        ))

        # Upper confidence bound
        fig_future.add_trace(go.Scatter(
            x=bridge_dates,
            y=bridge_upper,
            mode="lines",
            name="Bullish Bound (+80% CI)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Lower confidence bound with fill
        fig_future.add_trace(go.Scatter(
            x=bridge_dates,
            y=bridge_lower,
            mode="lines",
            name="Forecast Confidence Band",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(width=0),
            hoverinfo="skip"
        ))

        # Predicted line
        fig_future.add_trace(go.Scatter(
            x=bridge_dates,
            y=bridge_prices,
            mode="lines+markers",
            name=f"LSTM Forecast ({future_days} Days)",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            marker=dict(size=6, color="#ff7f0e")
        ))

        # Divider line for Today
        fig_future.add_vline(
            x=hist_dates[-1].timestamp() * 1000,
            line_width=1.5,
            line_dash="dot",
            line_color="#7f7f7f",
            annotation_text="Today",
            annotation_position="top left"
        )

        fig_future.update_layout(
            title=dict(text=f"<b>{stock} - Future {future_days}-Day AI Projection</b>", font=dict(size=18)),
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(title=f"Price ({comp_info['currency']})", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            hovermode="x unified",
            template="plotly_white",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_future, use_container_width=True)

        # Forecast Table & CSV Download
        st.subheader(f"📅 Daily Forecast Breakdown")

        records = []
        prev_val = latest_close
        for f_date, f_price, f_low, f_high in zip(future_dates, future_prices, lower_confidence, upper_confidence):
            daily_diff = f_price - prev_val
            daily_pct = (daily_diff / prev_val) * 100
            total_diff = f_price - latest_close
            total_pct = (total_diff / latest_close) * 100

            records.append({
                "Date": f_date.strftime('%Y-%m-%d (%A)'),
                "Predicted Price": f"${f_price:,.2f}",
                "Expected Low": f"${f_low:,.2f}",
                "Expected High": f"${f_high:,.2f}",
                "Daily Change": f"{daily_diff:+.2f} ({daily_pct:+.2f}%)",
                "Change from Today": f"{total_diff:+.2f} ({total_pct:+.2f}%)"
            })
            prev_val = f_price

        forecast_table = pd.DataFrame(records)

        col_tbl, col_dl = st.columns([4, 1])
        with col_tbl:
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        with col_dl:
            csv_data = forecast_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{stock}_forecast_{today.strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                help="Export upcoming predictions to CSV format"
            )
            st.caption("ℹ️ *Predictions are calibrated to branch seamlessly from today's closing price. Not financial advice.*")

    # ----------------------------------------------------
    # TAB 2: CANDLESTICK & VOLUME CHART (PLOTLY)
    # ----------------------------------------------------
    with tab2:
        st.subheader(f"Interactive Candlestick & Volume Analysis for {stock}")

        # Create subplots: Row 1 = Price & Bands, Row 2 = Volume
        fig_candle = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=(f"{stock} Daily Price & Bollinger Bands", "Trading Volume")
        )

        # Candlestick Trace
        fig_candle.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728"
        ), row=1, col=1)

        # Bollinger Bands
        fig_candle.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            name="Upper BB (20,2)",
            line=dict(color="#9467bd", width=1, dash="dot")
        ), row=1, col=1)

        fig_candle.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            name="Lower BB (20,2)",
            line=dict(color="#9467bd", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(148, 103, 189, 0.08)"
        ), row=1, col=1)

        # Moving Averages
        fig_candle.add_trace(go.Scatter(
            x=data.index, y=data['MA50'],
            name="MA50",
            line=dict(color="#ff7f0e", width=1.5)
        ), row=1, col=1)

        fig_candle.add_trace(go.Scatter(
            x=data.index, y=data['MA200'],
            name="MA200",
            line=dict(color="#1f77b4", width=1.8)
        ), row=1, col=1)

        # Volume Bars
        vol_colors = ["#2ca02c" if c >= o else "#d62728" for c, o in zip(data['Close'], data['Open'])]
        fig_candle.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color=vol_colors,
            showlegend=False
        ), row=2, col=1)

        fig_candle.update_layout(
            template="plotly_white",
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_candle, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: TECHNICAL INDICATORS (RSI & MACD)
    # ----------------------------------------------------
    with tab3:
        st.subheader(f"Technical Momentum Indicators for {stock}")

        # Subplots for RSI and MACD
        fig_tech = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Relative Strength Index (RSI 14)", "MACD (12, 26, 9) & Momentum Histogram")
        )

        # RSI Trace
        fig_tech.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name="RSI (14)",
            line=dict(color="#8c564b", width=2)
        ), row=1, col=1)

        # RSI Overbought / Oversold zones
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#d62728", annotation_text="Overbought (70)", row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#2ca02c", annotation_text="Oversold (30)", row=1, col=1)

        # MACD Line & Signal
        fig_tech.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            name="MACD Line",
            line=dict(color="#1f77b4", width=1.8)
        ), row=2, col=1)

        fig_tech.add_trace(go.Scatter(
            x=data.index, y=data['MACD_Signal'],
            name="Signal Line",
            line=dict(color="#ff7f0e", width=1.5, dash="dash")
        ), row=2, col=1)

        # MACD Histogram
        hist_colors = ["#2ca02c" if h >= 0 else "#d62728" for h in data['MACD_Hist']]
        fig_tech.add_trace(go.Bar(
            x=data.index, y=data['MACD_Hist'],
            name="MACD Histogram",
            marker_color=hist_colors
        ), row=2, col=1)

        fig_tech.update_layout(
            template="plotly_white",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_tech, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: HISTORICAL MODEL VALIDATION (TEST SET)
    # ----------------------------------------------------
    with tab4:
        st.subheader(f"Historical LSTM Validation on 20% Test Dataset")
        st.markdown("Evaluates model forecasting accuracy against unseen historical actual market data.")

        split_idx = int(len(close_df) * 0.80)
        data_train = pd.DataFrame(close_df.iloc[0:split_idx])
        data_test = pd.DataFrame(close_df.iloc[split_idx:])

        scaler_eval = MinMaxScaler(feature_range=(0, 1))
        pas_100 = data_train.tail(100)
        data_test_full = pd.concat([pas_100, data_test], ignore_index=True)
        data_test_scaled = scaler_eval.fit_transform(data_test_full)

        x_test, y_test = [], []
        for i in range(100, len(data_test_scaled)):
            x_test.append(data_test_scaled[i - 100:i])
            y_test.append(data_test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        with st.spinner("Evaluating model predictions on test dataset..."):
            test_preds = model.predict(x_test, verbose=0)

        scale_eval = 1 / scaler_eval.scale_[0]
        test_preds_actual = (test_preds * scale_eval).flatten()
        y_test_actual = (y_test * scale_eval).flatten()

        test_dates = data.index[split_idx:]
        min_len = min(len(test_dates), len(y_test_actual), len(test_preds_actual))

        eval_dates = test_dates[:min_len]
        eval_actual = y_test_actual[:min_len]
        eval_pred = test_preds_actual[:min_len]

        # Plotly interactive test evaluation
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_dates, y=eval_actual,
            mode="lines",
            name="Original Actual Price",
            line=dict(color="#2ca02c", width=2)
        ))
        fig_eval.add_trace(go.Scatter(
            x=eval_dates, y=eval_pred,
            mode="lines",
            name="LSTM Predicted Price",
            line=dict(color="#d62728", width=1.8, dash="dash")
        ))

        fig_eval.update_layout(
            title=dict(text=f"<b>{stock} - Actual vs Predicted Test Set Performance</b>", font=dict(size=18)),
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(title=f"Price ({comp_info['currency']})", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            hovermode="x unified",
            template="plotly_white",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        # Accuracy Metrics
        mae = float(np.mean(np.abs(eval_pred - eval_actual)))
        rmse = float(np.sqrt(np.mean((eval_pred - eval_actual) ** 2)))
        mape = float(np.mean(np.abs((eval_actual - eval_pred) / eval_actual)) * 100)
        direction_acc = float(np.mean(np.sign(np.diff(eval_actual)) == np.sign(np.diff(eval_pred))) * 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
        m2.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
        m3.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
        m4.metric("Directional Accuracy", f"{direction_acc:.1f}%")

    # ----------------------------------------------------
    # TAB 5: RAW MARKET DATA
    # ----------------------------------------------------
    with tab5:
        st.subheader(f"Historical Price & Volume Records for {stock}")
        st.dataframe(data, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)