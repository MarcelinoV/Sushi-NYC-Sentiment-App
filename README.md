# Sushi in NYC - A Sentiment Analysis

![alt_text](./readme_images/sushi_sentiment_pipeline.drawio.png "Project Pipeline")

Sentiment Analysis Web Dashboard focused on sushi restaurant reviews in New York City. Sourced from Google Places API, processed and modeled with dbt & Jupyter notebooks, and stored on Snowflake. The web app is then deployed as a docker container hosted on Azure App Services. 

Dashboard Link: https://sushi-nyc-sentiment-app-eng0h9h0h2fkcbb9.eastus2-01.azurewebsites.net/

## Code and Resources Used

**Python Version**: 3.12.9

**Packages**: dash, dash_bootstrap_components, plotly-express, pandas, snowflake-connector, wordcloud, numpy, nltk, re, transformers, scipy, requests

**Technologies**: SQL, Snowflake, DBT, Docker, Git, Github Actions, Microsoft Azure

**Data Source**: [Google Places API](https://developers.google.com/maps/documentation/places/web-service/text-search)

## Data Collection

I collected data about sushi restaurants in NYC from the Google Places API, an interface to the data that powers the popular Google Maps service. The Places API provides location data and imagery for establishments, geographic locations, and points of interest through various methods. I specifically used the the Text Search API, which is part of the Places API, to find places using text queries. I constructed my request to the API in a series of queries rather than a singular query like "sushi restaurants in New York City". Rather than the whole city, I queried by borough ("sushi restaurants in Queens", for example), and pulled pages of the response into two consumable dataframes. 

One for the location-related data, *raw_sushi_places*, and another for the reviews for each restaurant, *raw_sushi_reviews*. The dataframes are linked by a unique ID each location is assigned.

## Data Transformation & Feature Extraction

![alt_text](./readme_images/sushi_data_model.png "Project Data Model")

### Snowflake

Snowflake serves as the data warehouse for this application. Snowflake was chosen over other data warehouse services Databricks or a vendor-locked option like Amazon Redshift or Microsoft Azure Synapse Analytics because of its unique architecture that separates compute and storage. This allows for independent scaling, faster performance, and cost-eficiency, along with a fully managed, cloud-native approach.

Snowflake's capabilities also allows for further development of the data pipeline of the project, such as adding Snowpark Python UDFs, Python stored procedures, and tasks to facilitate and enhance data cleaning/processing.

### DBT

The raw data collected from the Google Places API is stored in Snowflake and is considered the *staging layer* in the the dbt lineage. It is then transformed in the *standardization layer* to a format where the restaurant's address is broken up into interpretable parts, like street address, city, borough, and zip code. The standardized table, *std_sushi_places*, is then used in conjunction with the raw data to generate the facts and dimensions in the *presentation layer*. These tables are named *sushi_places_dimension* and *sushi_place_reviews_fact*. The table that contains sentiment metrics of the reviews based on the text, *sushi_review_sentiment_fact*, is produced in a Jupyter notebook.

### Sentiment Models

The major feature of this application is to analyze a sushi restaurant through the sentiment of its reviews. This is valuable for not only informing on if a restaurant is performing well but to get a more intimate, on-the-ground insight of what the restaurant is like. For this, we utilize popular sentiment models like NLTK's VADER Sentiment Analysis and HuggingFace's RoBERTa Transformer models. The outputs of these models when ran on our review data make up the fields of the *sushi_review_sentiment_fact* table.

#### NLTK VADER

VADER, or Valence Aware Dictionary and sEntiment Reasoner, is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains. Being a rule-based model that uses a preset lexicon however, the model can fail to recognize sarcasm or context in language.

#### HuggingFace Transformers: RoBERTa

RoBERTa, which stands for Robustly Optimized BERT Pretraining Approach, is a powerful machine learning model pre-trained on a large corupus of text for NLP tasks. Specifically, its a variant of BERT, a neural network architecture designed for understanding the context of words in text. RoBERTa is optimized for improved performance and efficiency in various NLP tasks.

## Plotly Dash

## Deployment: Docker & Azure App Services

## CI/CD: Future Improvements

