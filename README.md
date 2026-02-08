# Tesla Stock Price Prediction System

##  Overview
This project is a time-series based stock price prediction system focused on Tesla (TSLA).  
The objective is to analyze historical stock data, extract meaningful technical indicators, and predict short-term price movement to support BUY / HOLD / SELL decisions.

The system emphasizes interpretability and robustness rather than black-box forecasting accuracy.

---

##  Objectives
- Predict short-term Tesla stock price trends
- Use technical indicators to capture market behavior
- Support decision-making (BUY / HOLD / SELL)
- Evaluate predictions using appropriate time-series metrics

---

##  Methodology

### 1. Data Collection
- Historical Tesla stock data (Open, High, Low, Close, Volume)
- Data sourced from financial APIs (e.g., yFinance)

---

### 2. Data Preprocessing
- Handling missing values
- Normalization of numerical features
- Time alignment of indicators
- Train-test split while preserving temporal order

---

### 3. Feature Engineering
The following technical indicators were used:

- SMA (Simple Moving Average)  
  Captures long-term trend direction

- RSI (Relative Strength Index)  
  Measures momentum and overbought/oversold conditions

- MACD (Moving Average Convergence Divergence)  
  Identifies trend reversals and momentum shifts

These indicators help transform raw price data into informative signals.

---

### 4. Model Design
- Time-series regression model
- Uses historical prices + engineered indicators
- Designed for short-term forecasting
- Focused on directional correctness rather than exact price prediction

---

### 5. Prediction Logic
- Model predicts future price for a given horizon
- Prediction converted into:
  - BUY → Expected upward movement
  - SELL → Expected downward movement
  - HOLD → Minor or uncertain movement

Thresholds are used to avoid noisy decisions.

---

##  Evaluation Strategy

The system is evaluated using:

- MAE (Mean Absolute Error)  
  Measures average prediction error

- RMSE (Root Mean Squared Error)  
  Penalizes large prediction errors

- Directional Accuracy  
  Measures correctness of predicted price movement direction

Evaluation focuses on decision quality, not just numeric accuracy.

---

##  Results
- Stable prediction behavior
- Improved performance compared to naive baselines
- High directional accuracy in trending markets
- Interpretable indicator-based decisions

---

##  Limitations
- Does not account for sudden news or macroeconomic events
- Performance may degrade in highly volatile markets
- Not intended for high-frequency trading
- Past performance does not guarantee future results

---

##  Future Improvements
- Integration of news sentiment analysis
- Advanced models (LSTM / Informer / Transformer)
- Risk-adjusted decision metrics
- Multi-stock generalization
- Portfolio-level optimization

---

##  Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Plotly
- Financial data APIs

---

##  Conclusion
This project demonstrates how technical indicators and time-series modeling can be used to create an interpretable stock prediction system that supports practical decision-making rather than relying solely on black-box models.

---

## 📎 Disclaimer
This project is for **educational purposes only** and should not be considered financial advice.
