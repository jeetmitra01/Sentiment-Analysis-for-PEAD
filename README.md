# Sentiment-Analysis-for-PEAD
The main question this project aims to answer is whether sentiment extracted from timestamped financial news headlines predicts abnormal stock returns after controlling for market movement, and does this signal translate into a tradable strategy?

## Dataset

I am using  the [Daily Financial News for 6000+ Stocks dataset](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download) from Kaggle for this project.
While the dataset itself is a bit old, it avoids a lot of the excess metadata in other datasets and only includes data that is necessary for this project.

## FinBERT backstory

FinBERT was built by a team at Prosus on top of BERT. <br>
If you're interested, [this article](https://medium.com/prosus-ai-tech-blog/finbert-financial-sentiment-analysis-with-bert-b277a3607101) walks through the process and motivations behind the project

## Data Preprocessing

Once we process the CSV file, we get a DataFrame with the following columns:

- `date`: The date of the news headline
- `headline`: The headline of the news
- `ticker`: The ticker of the stock
- `sentiment`: The sentiment of the headline
- `confidence`: The confidence of the sentiment

From here we can pull stock prices and calculate returns.

## Sentiment Analysis

As discussed earlier we are using FinBERT for sentiment analysis. Financial language is differeent from general language ad FinBERT is fine-tuned for this exact task, so it should perform better than a general purpose model.

## Analysis

the 'analysis.py' file contains the analysis code. It calculates returns by sentiment, conducts t-tests to see if the results are statistically significant, and generates plots to visualize the results.

## Results

It seems from intial analysis that the sentiment signal is not a good predictor of returns. PEAD has been a well documented phenomenon but from this analysis it seems that financial news sentiment is not a good predictor of returns. 

## Future Work

- Filter stocks by industry - to see if there are industries that might be more sensitive to sentiment.
- 

