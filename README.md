# Stock Price Prediction App

A web application for stock market analysis and future price prediction using deep learning (LSTM) and Streamlit.

## Live Link

Access the live web application here:
https://saisaran-stock-prediction.streamlit.app

## Features

- Real-Time Market Data: Fetches live historical stock data from Yahoo Finance.
- Multi-Day Future Forecasting: Uses a 4-layer LSTM neural network to predict upcoming trading days.
- Technical Indicators: Includes Moving Averages (MA50, MA100, MA200), Bollinger Bands, RSI (14), and MACD.
- Interactive Visualizations: Built with Plotly for interactive candlestick, volume, and prediction charts.
- CSV Export: Download upcoming price forecast tables as CSV files.

## Project Structure

- app.py: Main Streamlit web application.
- train_model.py: Training script for the LSTM neural network.
- Stock Predictions Model.keras: Pre-trained LSTM model file.
- requirements.txt: Python package dependencies.
- Stock_Market_Prediction_Model_Creation.ipynb: Jupyter notebook for model exploration.

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/saisaran-m/stockprice-prediction.git
cd stockprice-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to:
```text
http://localhost:8501
```

## How It Works

1. Enter any stock symbol (such as GOOG, AAPL, MSFT, TSLA).
2. Select your desired date range and forecast horizon.
3. The app downloads the latest market data, computes technical indicators, and runs the LSTM model to generate future price estimates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

