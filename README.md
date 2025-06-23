# 📈 Event-Driven Sentiment: Analyzing Fed Statements with FinBERT

This project analyzes whether the **tone of U.S. Federal Reserve policy statements** can explain or predict **short-term movements in financial markets**, specifically:

- 📉 **S&P 500 index** returns  
- 💵 **10-year U.S. Treasury yields**

To measure sentiment, we use the **FinBERT** transformer model from Hugging Face, fine-tuned on financial texts.

---

## 🔍 Main Results

### 💬 Sentiment vs. S&P 500 Change (Next Day)

![S&P 500 Plot](results/plot_sentiment_sp500.png)

> The relationship is weak and noisy. Most statements are neutral, and there's no clear upward or downward trend.

---

### 💬 Sentiment vs. 10-Year Treasury Yield Change

![10Y Yield Plot](results/plot_sentiment_tnx.png)

> No meaningful correlation observed. Market reaction may reflect expectations or occur intraday.

---

### 📄 Regression Output: S&P 500 ~ Sentiment Score

