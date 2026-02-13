import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")

st.title("📊 Trader Behavior & Sentiment Analysis Dashboard")
st.markdown("Analysis of Trader Performance vs Fear & Greed Index")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    trades = pd.read_csv("historical_data.csv")
    sentiment = pd.read_csv("fear_greed_index.csv")
    return trades, sentiment

trades, sentiment = load_data()

# -----------------------------
# Data Cleaning
# -----------------------------
trades["Timestamp IST"] = pd.to_datetime(
    trades["Timestamp IST"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

trades["date"] = trades["Timestamp IST"].dt.date
sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date

trades["is_win"] = trades["Closed PnL"] > 0
trades["is_long"] = trades["Side"].str.lower().str.contains("long")

# -----------------------------
# Daily Aggregation
# -----------------------------
daily = trades.groupby("date").agg(
    daily_pnl=("Closed PnL", "sum"),
    win_rate=("is_win", "mean"),
    trades_per_day=("Closed PnL", "count"),
    avg_trade_size=("Size USD", "mean"),
    avg_leverage=("Start Position", "mean"),
    long_ratio=("is_long", "mean")
).reset_index()

data = daily.merge(
    sentiment[["date", "classification", "value"]],
    on="date",
    how="inner"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=data["classification"].unique(),
    default=data["classification"].unique()
)

filtered_data = data[data["classification"].isin(sentiment_filter)]

# -----------------------------
# Summary Metrics
# -----------------------------
st.subheader("📌 Summary Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Average Daily PnL", round(filtered_data["daily_pnl"].mean(), 2))
col2.metric("Average Win Rate", round(filtered_data["win_rate"].mean(), 2))
col3.metric("Average Trades/Day", round(filtered_data["trades_per_day"].mean(), 2))

# -----------------------------
# FIXED GRAPHS SECTION
# -----------------------------
st.subheader("📈 Performance & Behavior Charts")

if not filtered_data.empty:

    # Daily PnL
    fig1, ax1 = plt.subplots()
    filtered_data.boxplot(column="daily_pnl", by="classification", ax=ax1)
    plt.suptitle("")
    ax1.set_title("Daily PnL by Sentiment")
    st.pyplot(fig1)
    plt.close(fig1)

    # Win Rate
    fig2, ax2 = plt.subplots()
    filtered_data.boxplot(column="win_rate", by="classification", ax=ax2)
    plt.suptitle("")
    ax2.set_title("Win Rate by Sentiment")
    st.pyplot(fig2)
    plt.close(fig2)

    # Trades Per Day
    fig3, ax3 = plt.subplots()
    filtered_data.boxplot(column="trades_per_day", by="classification", ax=ax3)
    plt.suptitle("")
    ax3.set_title("Trades Per Day by Sentiment")
    st.pyplot(fig3)
    plt.close(fig3)

    # Leverage
    fig4, ax4 = plt.subplots()
    filtered_data.boxplot(column="avg_leverage", by="classification", ax=ax4)
    plt.suptitle("")
    ax4.set_title("Average Leverage by Sentiment")
    st.pyplot(fig4)
    plt.close(fig4)

else:
    st.warning("No data available for selected filters.")

# -----------------------------
# STRATEGY RECOMMENDATIONS SECTION
# -----------------------------
st.subheader("📊 Strategy Recommendations (Part C)")

if not filtered_data.empty:

    avg_freq = filtered_data["trades_per_day"].mean()
    avg_lev = filtered_data["avg_leverage"].mean()
    avg_sentiment = filtered_data["value"].mean()

    st.markdown("### 🎯 Rule 1 — Control Overtrading")

    if avg_freq > daily["trades_per_day"].median():
        st.warning(
            "Trade frequency is elevated. Consider reducing daily trade count by 20–30% to avoid overtrading losses."
        )
    else:
        st.success(
            "Trade frequency is within healthy range. Maintain disciplined execution."
        )

    st.markdown("### ⚡ Rule 2 — Leverage Discipline Based on Sentiment")

    if avg_sentiment > 70:
        st.warning(
            "Market is in Greed regime. Reduce leverage by 15–25% to protect against volatility spikes."
        )
    elif avg_sentiment < 30:
        st.warning(
            "Market is in Extreme Fear regime. Use controlled leverage and avoid aggressive scaling."
        )
    else:
        st.success(
            "Market sentiment is Neutral. Moderate leverage can be maintained."
        )

# -----------------------------
# Predictive Model
# -----------------------------
st.subheader("🤖 Next-Day Profitability Prediction")

model_data = data.sort_values("date").copy()
model_data["next_day_pnl"] = model_data["daily_pnl"].shift(-1)
model_data["target"] = (model_data["next_day_pnl"] > 0).astype(int)
model_data = model_data.dropna()

features = ["value", "win_rate", "trades_per_day", "avg_leverage", "long_ratio"]

X = model_data[features]
y = model_data["target"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X, y)

st.markdown("### Enter Current Market Conditions")

input_value = st.number_input("Sentiment Value", value=50)
input_win = st.number_input("Win Rate", value=0.4)
input_trades = st.number_input("Trades Per Day", value=100)
input_lev = st.number_input("Average Leverage", value=1000.0)
input_long = st.number_input("Long Ratio", value=0.5)

if st.button("Predict Next Day Outcome"):
    input_df = pd.DataFrame(
        [[input_value, input_win, input_trades, input_lev, input_long]],
        columns=features
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"📈 Model predicts NEXT DAY PROFITABLE (Confidence: {round(probability*100,2)}%)")
    else:
        st.error(f"📉 Model predicts NEXT DAY LOSS (Confidence: {round((1-probability)*100,2)}%)")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📊 Model Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

fig5, ax5 = plt.subplots()
importance.plot(kind="bar", ax=ax5)
ax5.set_title("Feature Importance")
st.pyplot(fig5)
plt.close(fig5)

st.markdown("🚀 Dashboard Complete — Sentiment + Behavior + Strategy + Prediction")
