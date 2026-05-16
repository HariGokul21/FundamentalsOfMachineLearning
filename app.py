from datetime import date, timedelta


import nltk
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MarketMood - Sentiment-Based Next-Day Stock Trend Prediction",
    layout="wide",
)



# -----------------------------
# Basic settings
# -----------------------------
API_KEY = "d7nq20hr01qm3637cnf0d7nq20hr01qm3637cnfg"


RANGE_OPTIONS = {
    "1 month": "1mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}


DEFAULT_WATCHLIST = ["MSFT", "NVDA", "AAPL", "RELIANCE.NS"]



# -----------------------------
# Formatting helpers
# -----------------------------
def show_value(value, value_type="text"):
    if value is None or value == "" or value == "None":
        return "N/A"


    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass


    if value_type == "money":
        try:
            return f"${float(value):,.2f}"
        except Exception:
            return "N/A"


    if value_type == "large_number":
        try:
            value = float(value)
            if value >= 1_000_000_000_000:
                return f"${value / 1_000_000_000_000:.2f}T"
            if value >= 1_000_000_000:
                return f"${value / 1_000_000_000:.2f}B"
            if value >= 1_000_000:
                return f"${value / 1_000_000:.2f}M"
            return f"${value:,.0f}"
        except Exception:
            return "N/A"


    if value_type == "number":
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return "N/A"


    if value_type == "integer":
        try:
            return f"{int(value):,}"
        except Exception:
            return "N/A"


    if value_type == "percent":
        try:
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return "N/A"


    return str(value)



def make_range(low, high):
    low_text = show_value(low, "money")
    high_text = show_value(high, "money")


    if low_text == "N/A" or high_text == "N/A":
        return "N/A"


    return f"{low_text} - {high_text}"



# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_resource
def get_sentiment_analyzer():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()



@st.cache_data(ttl=3600)
def fetch_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not isinstance(info, dict) or len(info) <= 1:
            return {}
        return info
    except Exception:
        return {}



@st.cache_data(ttl=3600)
def validate_ticker(ticker):
    info = fetch_company_info(ticker)


    if not info:
        return False, {}, "Ticker data could not be loaded from Yahoo Finance."


    symbol = info.get("symbol") or ticker
    quote_type = info.get("quoteType", "Unknown")
    exchange = info.get("fullExchangeName") or info.get("exchange") or "Unknown"


    if quote_type in ["EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY"]:
        return True, info, f"{symbol} verified on {exchange}."


    return False, info, f"'{ticker}' was found, but the instrument type is not supported. Exchange: {exchange}, Type: {quote_type}."



@st.cache_data(ttl=3600)
def fetch_stock_prices(ticker, period="6mo"):
    try:
        prices = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )


        if prices.empty:
            return pd.DataFrame()


        prices = prices.reset_index()


        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [col[0] for col in prices.columns]


        prices = prices.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )


        required_columns = ["date", "open", "high", "low", "close", "volume"]
        available_columns = [col for col in required_columns if col in prices.columns]


        if "date" not in available_columns or "close" not in available_columns:
            return pd.DataFrame()


        prices = prices[available_columns].dropna()
        prices["date"] = pd.to_datetime(prices["date"])


        return prices


    except Exception:
        return pd.DataFrame()



@st.cache_data(ttl=3600)
def fetch_news(ticker):
    today = date.today()
    start_date = today - timedelta(days=14)


    if API_KEY == "YOUR_KEY_HERE" or not API_KEY:
        return pd.DataFrame(), "No Finnhub API key is set. Live sentiment cannot be loaded."


    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start_date.isoformat(),
        "to": today.isoformat(),
        "token": API_KEY,
    }


    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news_items = response.json()


        rows = []
        for item in news_items[:20]:
            headline = item.get("headline")
            published_time = item.get("datetime")


            if headline:
                rows.append(
                    {
                        "date": pd.to_datetime(published_time, unit="s").date()
                        if published_time
                        else today,
                        "source": item.get("source", "Finnhub"),
                        "headline": headline,
                    }
                )


        if not rows:
            return pd.DataFrame(), f"No live headlines were found for {ticker} over the last 14 days."


        return pd.DataFrame(rows), f"Recent live headlines loaded for {ticker}."


    except Exception:
        return pd.DataFrame(), f"Live news could not be loaded for {ticker} due to an API error."



@st.cache_data(ttl=3600)
def build_watchlist_table(tickers):
    rows = []


    for symbol in tickers:
        info = fetch_company_info(symbol)
        if not info:
            rows.append(
                {
                    "Ticker": symbol,
                    "Name": "N/A",
                    "Exchange": "N/A",
                    "Price": "N/A",
                    "Change %": "N/A",
                    "Market Cap": "N/A",
                }
            )
            continue


        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose")
        change_pct = None
        if current_price not in [None, ""] and previous_close not in [None, 0, ""]:
            try:
                change_pct = (float(current_price) / float(previous_close) - 1) * 100
            except Exception:
                change_pct = None


        rows.append(
            {
                "Ticker": symbol,
                "Name": info.get("shortName") or info.get("longName") or symbol,
                "Exchange": info.get("fullExchangeName") or info.get("exchange") or "N/A",
                "Price": show_value(current_price, "money"),
                "Change %": f"{change_pct:.2f}%" if change_pct is not None else "N/A",
                "Market Cap": show_value(info.get("marketCap"), "large_number"),
            }
        )


    return pd.DataFrame(rows)



# -----------------------------
# Data helpers
# -----------------------------
def score_sentiment(news_df):
    if news_df.empty or "headline" not in news_df.columns:
        return pd.DataFrame()


    analyzer = get_sentiment_analyzer()
    scored = news_df.copy()


    scored["sentiment"] = scored["headline"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )


    scored["sentiment_label"] = pd.cut(
        scored["sentiment"],
        bins=[-1, -0.05, 0.05, 1],
        labels=["Negative", "Neutral", "Positive"],
    )


    return scored



def build_model_data(price_df, avg_sentiment):
    if price_df.empty or len(price_df) < 40:
        return pd.DataFrame()


    data = price_df.copy()


    if "close" not in data.columns or "volume" not in data.columns:
        return pd.DataFrame()


    data["return_1d"] = data["close"].pct_change()
    data["volume_change"] = data["volume"].pct_change()
    data["sentiment"] = avg_sentiment
    data["next_close"] = data["close"].shift(-1)
    data["target_up"] = (data["next_close"] > data["close"]).astype(int)


    data = data.dropna().reset_index(drop=True)
    return data



def train_models(model_df):
    if model_df.empty or len(model_df) < 40:
        return None, None, None


    if model_df["target_up"].nunique() < 2:
        return None, None, None


    features = ["sentiment", "return_1d", "volume_change"]
    X = model_df[features]
    y = model_df["target_up"]


    split_index = int(len(model_df) * 0.75)


    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    if len(X_train) < 10 or len(X_test) < 5 or y_train.nunique() < 2:
        return None, None, None


    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)


    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)


    comparison = pd.DataFrame(
        {
            "Model": ["Simple Baseline", "Logistic Regression"],
            "Accuracy": [
                accuracy_score(y_test, baseline_pred),
                accuracy_score(y_test, logistic_pred),
            ],
        }
    )


    latest_features = X.tail(1)
    probability_up = logistic.predict_proba(latest_features)[0][1]


    return logistic, comparison, probability_up



def get_prediction_from_current_data(ticker, avg_sentiment):
    prices = fetch_stock_prices(ticker, period="6mo")


    if prices.empty:
        return None, None, "Price data is unavailable."


    model_df = build_model_data(prices, avg_sentiment)


    if model_df.empty or len(model_df) < 40:
        return None, None, "Not enough data to train the model."


    model, comparison, probability_up = train_models(model_df)


    if model is None:
        return None, None, "The model could not be trained safely with the available data."


    return probability_up, comparison, "Prediction ready."



# -----------------------------
# Session state
# -----------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()



# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("MarketMood")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "News & Sentiment", "Model Insights"],
)


ticker_input = st.sidebar.text_input(
    "Enter any ticker from any exchange",
    value="MSFT",
    help="Examples: MSFT, NVDA, RELIANCE.NS, 7203.T, VOD.L, TSLA, ^NSEI, BTC-USD",
)
ticker = ticker_input.strip().upper()


st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")
watchlist_input = st.sidebar.text_input(
    "Add ticker to watchlist",
    value="",
    placeholder="Type symbol and click add",
)


col_add, col_reset = st.sidebar.columns(2)
if col_add.button("Add"):
    new_symbol = watchlist_input.strip().upper()
    if not new_symbol:
        st.sidebar.warning("Enter a ticker first.")
    elif new_symbol in st.session_state.watchlist:
        st.sidebar.info("Ticker already exists in watchlist.")
    else:
        is_valid_watch, _, _ = validate_ticker(new_symbol)
        if is_valid_watch:
            st.session_state.watchlist.append(new_symbol)
            st.sidebar.success(f"Added {new_symbol}.")
        else:
            st.sidebar.error("Ticker could not be verified.")


if col_reset.button("Reset"):
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
    st.sidebar.success("Watchlist reset.")


if st.session_state.watchlist:
    selected_watch = st.sidebar.selectbox(
        "Open from watchlist",
        st.session_state.watchlist,
        index=0,
    )
    if st.sidebar.button("Use selected watchlist ticker"):
        ticker = selected_watch



# -----------------------------
# Main Content Handling
# -----------------------------
st.title("MarketMood - A Sentiment-Based Approach to Next-Day Stock Trend Prediction")
st.write(
    "This app checks whether recent news sentiment around a ticker is linked to next-day movement and accepts symbols from any exchange that Yahoo Finance supports."
)


if not ticker:
    st.warning("Please type a valid ticker symbol in the sidebar.")
    st.stop()


is_valid, info, validation_message = validate_ticker(ticker)


if not is_valid:
    if info:
        st.error(validation_message)
    else:
        st.error(f"Could not load data for '{ticker}'. Please ensure it is a valid Yahoo Finance symbol.")
    st.stop()


st.success(validation_message)


watchlist_df = build_watchlist_table(st.session_state.watchlist)


with st.expander("Watchlist overview", expanded=True):
    st.dataframe(watchlist_df, use_container_width=True, hide_index=True)



# -----------------------------
# Home page
# -----------------------------
if page == "Home":
    st.header(f"{ticker} Home")


    st.subheader("Company / instrument description")
    st.write(info.get("longBusinessSummary", "No official description found for this symbol."))


    st.subheader("Instrument details")


    instrument_name = info.get("longName") or info.get("shortName") or ticker
    exchange_name = info.get("fullExchangeName") or info.get("exchange")
    quote_type = info.get("quoteType")
    sector = info.get("sector")
    industry = info.get("industry")
    currency = info.get("currency")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose")
    open_price = info.get("open")
    day_range = make_range(info.get("dayLow"), info.get("dayHigh"))
    year_range = make_range(info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh"))
    volume = info.get("volume")
    average_volume = info.get("averageVolume")
    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")
    eps = info.get("trailingEps")
    dividend_yield = info.get("dividendYield")
    beta = info.get("beta")
    target_estimate = info.get("targetMeanPrice")


    details = pd.DataFrame(
        {
            "Metric": [
                "Name",
                "Symbol",
                "Exchange",
                "Type",
                "Currency",
                "Sector",
                "Industry",
                "Current price",
                "Previous close",
                "Open",
                "Day range",
                "52 week range",
                "Volume",
                "Average volume",
                "Market cap",
                "PE ratio",
                "EPS",
                "Dividend yield",
                "Beta",
                "1 year target estimate",
            ],
            "Value": [
                show_value(instrument_name),
                show_value(ticker),
                show_value(exchange_name),
                show_value(quote_type),
                show_value(currency),
                show_value(sector),
                show_value(industry),
                show_value(current_price, "money"),
                show_value(previous_close, "money"),
                show_value(open_price, "money"),
                day_range,
                year_range,
                show_value(volume, "integer"),
                show_value(average_volume, "integer"),
                show_value(market_cap, "large_number"),
                show_value(pe_ratio, "number"),
                show_value(eps, "number"),
                show_value(dividend_yield, "percent"),
                show_value(beta, "number"),
                show_value(target_estimate, "money"),
            ],
        }
    )


    st.dataframe(details, use_container_width=True, hide_index=True)


    st.subheader("Simple price viewer")


    selected_range = st.selectbox(
        "Choose chart time range",
        list(RANGE_OPTIONS.keys()),
        index=4,
    )


    prices = fetch_stock_prices(ticker, period=RANGE_OPTIONS[selected_range])


    if prices.empty:
        st.error("No price data found. Try again later or choose another ticker.")
    else:
        st.write(f"Showing {selected_range} of daily price data for {ticker}.")


        latest_close = prices["close"].iloc[-1]
        first_close = prices["close"].iloc[0]
        period_return = (latest_close / first_close - 1) * 100


        col1, col2, col3 = st.columns(3)
        col1.metric("Latest close", show_value(latest_close, "money"))
        col2.metric("First close in range", show_value(first_close, "money"))
        col3.metric("Return in selected range", f"{period_return:.1f}%")


        line_fig = px.line(
            prices,
            x="date",
            y="close",
            title=f"{ticker} closing price",
            labels={"date": "Date", "close": "Closing price"},
        )
        st.plotly_chart(line_fig, use_container_width=True)


        clean_table = prices.copy()
        clean_table["date"] = clean_table["date"].dt.date
        clean_table = clean_table.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )


        st.write("Recent price data:")
        st.dataframe(
            clean_table.tail(15),
            use_container_width=True,
            hide_index=True,
        )



# -----------------------------
# News & Sentiment page
# -----------------------------
elif page == "News & Sentiment":
    st.header(f"News & Sentiment for {ticker}")


    news_df, news_message = fetch_news(ticker)


    if news_df.empty:
        st.warning(news_message)
        st.info("Because no live news headlines were retrieved for this symbol, sentiment parsing cannot be calculated.")
    else:
        scored_news = score_sentiment(news_df)
        st.info(news_message)


        st.subheader("Recent headlines")
        st.dataframe(
            scored_news[["date", "source", "headline", "sentiment", "sentiment_label"]],
            use_container_width=True,
            hide_index=True,
        )


        avg_sentiment = scored_news["sentiment"].mean()
        positive_count = (scored_news["sentiment"] > 0.05).sum()
        negative_count = (scored_news["sentiment"] < -0.05).sum()
        neutral_count = (
            (scored_news["sentiment"] >= -0.05) & (scored_news["sentiment"] <= 0.05)
        ).sum()


        st.subheader("Sentiment summary")


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average sentiment score", f"{avg_sentiment:.3f}")
        col2.metric("Positive headlines", int(positive_count))
        col3.metric("Neutral headlines", int(neutral_count))
        col4.metric("Negative headlines", int(negative_count))


        sentiment_fig = px.histogram(
            scored_news,
            x="sentiment",
            color="sentiment_label",
            title="How positive or negative are the headlines?",
            labels={
                "sentiment": "Sentiment score",
                "sentiment_label": "Sentiment type",
            },
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)


        probability_up, comparison, _ = get_prediction_from_current_data(
            ticker, avg_sentiment
        )


        st.subheader("My prediction for next-day movement")


        if probability_up is None:
            st.write(
                "The app cannot estimate next-day movement right now because there is not enough valid ML training data."
            )
        else:
            probability_down = 1 - probability_up


            if probability_up >= 0.50:
                st.write(
                    f"Based on the current model, the stock has an estimated {probability_up:.1%} chance of moving up tomorrow."
                )
            else:
                st.write(
                    f"Based on the current model, the stock has an estimated {probability_down:.1%} chance of moving down tomorrow."
                )



# -----------------------------
# Model Insights page
# -----------------------------
elif page == "Model Insights":
    st.header(f"Model Insights for {ticker}")


    news_df, _ = fetch_news(ticker)


    avg_sentiment = 0.0
    if not news_df.empty:
        scored_news = score_sentiment(news_df)
        avg_sentiment = scored_news["sentiment"].mean()
    else:
        st.warning("No headline data available to calculate active news sentiment. Running the models with a default neutral score (0.0).")


    prices = fetch_stock_prices(ticker, period="6mo")


    if prices.empty:
        st.error("No price data returned by yfinance. The model cannot be trained.")
        st.stop()


    model_df = build_model_data(prices, avg_sentiment)


    if model_df.empty or len(model_df) < 40:
        st.error("Not enough price data to train the simple model.")
        st.stop()


    model, comparison, probability_up = train_models(model_df)


    if model is None or comparison is None or probability_up is None:
        st.error("Not enough valid data variance to train Logistic Regression safely.")
        st.stop()


    probability_down = 1 - probability_up


    st.subheader("Next-day movement estimate")


    col1, col2 = st.columns(2)
    col1.metric("Chance of moving up", f"{probability_up:.1%}")
    col2.metric("Chance of moving down", f"{probability_down:.1%}")


    if probability_up > 0.55:
        st.success("The model leans slightly upward.")
    elif probability_up < 0.45:
        st.error("The model leans slightly downward.")
    else:
        st.info("The model is uncertain.")


    st.subheader("Model comparison")
    st.write("This table compares a very simple baseline model with Logistic Regression.")


    readable_comparison = comparison.copy()
    readable_comparison["Accuracy"] = readable_comparison["Accuracy"].apply(
        lambda x: f"{x:.1%}"
    )


    st.dataframe(
        readable_comparison,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader("Recent price movement")


    recent_prices = prices.copy()
    recent_prices["daily_change"] = recent_prices["close"].pct_change() * 100
    recent_prices = recent_prices.dropna().tail(30)


    price_fig = px.line(
        recent_prices,
        x="date",
        y="close",
        title="Closing price over the last 30 trading days",
        labels={"date": "Date", "close": "Closing price"},
    )
    st.plotly_chart(price_fig, use_container_width=True)


    st.subheader("How often did the stock move up or down?")


    movement_counts = (
        model_df["target_up"]
        .map({1: "Up next day", 0: "Down next day"})
        .value_counts()
        .reset_index()
    )
    movement_counts.columns = ["Movement", "Days"]


    movement_fig = px.bar(
        movement_counts,
        x="Movement",
        y="Days",
        title="Recent next-day movement count",
        labels={"Movement": "Next-day movement", "Days": "Number of days"},
    )
    st.plotly_chart(movement_fig, use_container_width=True)


    st.subheader("Simple explanation")
    st.write(
        """
        The model uses three beginner-friendly inputs:
        - recent news sentiment
        - previous daily stock return
        - change in trading volume


        It compares a baseline model with Logistic Regression and then estimates whether the next day is more likely to be up or down.
        """
    )