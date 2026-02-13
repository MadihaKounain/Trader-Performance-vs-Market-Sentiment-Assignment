# 📊 Trader Behavior & Sentiment Analysis Dashboard

## 📌 Project Overview

This project analyzes the relationship between trader behavior and the
Crypto Fear & Greed Index. It explores how sentiment regimes influence:

-   Trader profitability (PnL)
-   Win rate
-   Trade frequency
-   Leverage usage
-   Long/short bias

The project includes: - Trader segmentation analysis - A predictive
model for next-day profitability (\~82% accuracy) - A Streamlit
dashboard for interactive exploration - Actionable strategy
recommendations

------------------------------------------------------------------------

# ⚙️ Setup Instructions

## 1️⃣ Project Structure

Ensure the folder contains:

-   app.py
-   historical_data.csv
-   fear_greed_index.csv

------------------------------------------------------------------------

## 2️⃣ Create Virtual Environment (Recommended)

``` bash
python -m venv venv
```

Activate:

Windows:

``` bash
venv\Scripts\activate
```

Mac/Linux:

``` bash
source venv/bin/activate
```

------------------------------------------------------------------------

## 3️⃣ Install Dependencies

``` bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

------------------------------------------------------------------------

## 4️⃣ Run the Dashboard

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# 📊 Output Charts & Tables

The dashboard includes:

### 📈 Performance Charts

-   Daily PnL by Sentiment (Boxplot)
-   Win Rate by Sentiment
-   Trades per Day by Sentiment
-   Average Leverage by Sentiment

### 📌 Summary Metrics

-   Average Daily PnL
-   Average Win Rate
-   Average Trades per Day

### 🤖 Predictive Model

-   Next-day profitability prediction
-   Confidence probability
-   Feature importance chart

### 📊 Strategy Recommendations

Dynamic rule-based guidance based on: - Sentiment regime - Trade
frequency - Leverage level

------------------------------------------------------------------------

# 📝 Project Summary (Methodology, Insights, Strategy)

## Methodology

1.  Data Cleaning
    -   Loaded trade-level data and Fear & Greed Index\
    -   Converted timestamps and aligned datasets at daily frequency\
    -   Verified no missing values or duplicates
2.  Feature Engineering\
    Created daily metrics:
    -   Daily PnL\
    -   Win rate\
    -   Trades per day\
    -   Average leverage\
    -   Long ratio
3.  Sentiment-Based Analysis
    -   Compared performance across Fear, Neutral, and Greed regimes\
    -   Analyzed behavioral changes across regimes
4.  Trader Segmentation
    -   High vs Low Leverage traders\
    -   Frequent vs Infrequent traders\
    -   Consistent vs Inconsistent traders
5.  Predictive Modeling
    -   Built Random Forest model\
    -   Used sentiment + behavioral features\
    -   Achieved \~82% accuracy\
    -   Identified sentiment and leverage as strongest predictors

------------------------------------------------------------------------

## Key Insights

1️⃣ Sentiment Drives Performance\
The Fear & Greed Index value was the most important predictive feature.
Market regime significantly impacts next-day profitability.

2️⃣ Overtrading Reduces Edge\
Infrequent traders outperformed frequent traders significantly. Higher
trade frequency was associated with lower average PnL.

3️⃣ Leverage Matters More Than Win Rate\
Leverage had higher predictive power than historical win rate. Risk
exposure is more important than historical success rate.

4️⃣ Direction Alone Is Weak\
Long/short bias alone has limited predictive power compared to sentiment
and leverage.

------------------------------------------------------------------------

## Strategy Recommendations (Actionable Output)

### 🎯 Rule 1 --- Control Overtrading

During Greed regimes: - Reduce trade frequency by 20--30% - Avoid
chasing momentum - Focus on high-conviction setups

Rationale: Frequent traders underperformed significantly.

------------------------------------------------------------------------

### ⚡ Rule 2 --- Dynamic Leverage Management

During Extreme Greed or Extreme Fear: - Reduce leverage by 15--25% -
Avoid aggressive scaling - Maintain strict risk limits

Rationale: Leverage is a major predictive driver and amplifies
volatility risk.

------------------------------------------------------------------------

# 🚀 Project Strength

✔ Combines behavioral finance + machine learning\
✔ Connects macro sentiment with trading behavior\
✔ Includes segmentation analysis\
✔ Includes predictive modeling\
✔ Produces actionable strategy output\
✔ Delivered via interactive dashboard
