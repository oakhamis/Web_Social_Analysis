## STUDYING THE 2021 AMERICAN ASIAN PACIFIC ISLANDER MOVEMENT USING TWITTER & NEWS DATA

This repository contains analysis scripts that explore the 2021 American Asian Pacific Islander movement using Twitter and news data. The two primary scripts in this repository delve deep into the sentiment, interactions, and trends associated with this movement.

## News Data Analysis

### Overview
The first script focuses on analyzing news data related to the American Asian Pacific Islander movement. It offers functionalities like web scraping, data preprocessing, subjectivity analysis, sentiment analysis, and topic modeling.

### Features
- **Web Scraping**: Extracts news articles using Beautiful Soup.
- **Data Cleaning**: Strips HTML tags, special characters, and unwanted whitespace.
- **Tokenization**: Breaks the news content into individual words.
- **Sentiment Analysis**: Uses TextBlob to determine the sentiment of each article.
- **Subjectivity Analysis**: Uses TextBlob to determine the sentiment of each article.
- **Topic Modeling**: Uses LDA (Latent Dirichlet Allocation) to identify the major topics covered in the news articles.

## Twitter Data Analysis

### Overview
The second script dives into the Twitter data corresponding to the movement. It fetches tweets, performs sentiment analysis, visualizes data, and explores the interaction network of the users.

### Features
- **Fetching Tweets**: Uses Tweepy's Cursor object to fetch relevant tweets.
- **Data Pre-processing**: Removes duplicates, handles missing values, and extracts hashtags and devices.
- **Heatmap Creation**: Maps user locations to generate a heatmap of tweet concentrations.
- **Sentiment Analysis**: Classifies tweets as Positive, Negative, or Neutral.
- **Graph Analysis**: Constructs an interaction graph based on retweets, mentions, and replies. Computes basic graph metrics and visualizes the graph.
- **Community Detection**: Implements the Louvain algorithm to detect communities within the graph.
- **Centrality Measures**: Calculates Degree Centrality, Closeness Centrality, and Betweenness Centrality.

## Dependencies
Before running the scripts, ensure you have the following libraries installed.

## Usage
1. Ensure you have the necessary credentials for the Twitter API.
2. Set the working directory and paths as required.
3. Run the scripts.

## Contributing
If you'd like to contribute, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.
