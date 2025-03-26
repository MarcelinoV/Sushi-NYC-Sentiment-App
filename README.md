# Sushi in NYC - A Sentiment Analysis

![alt_text](./readme_images/sushi_sentiment_pipeline.drawio.png "Project Pipeline")

Sentiment Analysis data application focused on sushi restaurant reviews in New York City. sourced from Google Places API, processed and modeled with dbt & Jupyter notebooks, and stored on Snowflake. The web app is then deployed as a docker container hosed on Azure App Services. 

Dashboard Link: https://sushi-nyc-sentiment-app-eng0h9h0h2fkcbb9.eastus2-01.azurewebsites.net/

## Code and Resources Used

**Python Version**: 3.12.9

**Packages**: dash, dash_bootstrap_components, plotly-express, pandas, snowflake-connector, wordcloud, numpy, nltk, re, transformers, scipy, requests

**Technologies**: SQL, Snowflake, DBT, Docker, Git, Github Actions, Microsoft Azure

**Data Source**: [Google Places API](https://developers.google.com/maps/documentation/places/web-service/text-search)

## Data Collection

I collected data about sushi restaurants in NYC from the Google Places API, an interface to the data that powers the popular Google Maps service. The Places API provides location data and imagery for establishments, geographic locations, and points of interest through various methods. I specifically used the the Text Search API, which is part of the Places API, to find places using text queries. I constructed my request to the API in a series of queries rather than a singular query like "sushi restaurants in New York City". Rather than the whole city, I queried by borough ("sushi restaurants in Queens", for example), and pulled pages of the response into consumable dataframes. 

## Analytics Engineering: Data Transformation & Feature Extraction

![alt_text](./readme_images/sushi_data_model.png "Project Data Model")

### Snowflake

### DBT

### Sentiment Models

#### NLTK VADER

#### HuggingFace Transformers: RoBERTa

## Presentation: Plotly Dash

## Deployment: Docker & Azure App Services

## CI/CD: Future Improvements

