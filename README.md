# Sentiment-Analysis-for-PEAD
The main question this project aims to answer is whether sentiment extracted from timestamped financial news headlines predicts abnormal stock returns after controlling for market movement, and does this signal translate into a tradable strategy?

## Dataset

I am using  the [Daily Financial News for 6000+ Stocks dataset](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download) from Kaggle for this project.
While the dataset itself is a bit old, it avoids a lot of the useless metadata in other datasets and only includes data that is necessary for this project.

## FinBERT backstory

FinBERT was built by a team at Prosus on top of BERT. <br>
If you're interested, [this article](https://medium.com/prosus-ai-tech-blog/finbert-financial-sentiment-analysis-with-bert-b277a3607101) walks through the process and motivations behind the projext
