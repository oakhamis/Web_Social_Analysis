"""
This code performs Tweet Extraction & Analysis
"""

# Importing necessary libraries and packages
import networkx as nx
from cdlib import algorithms, viz
from matplotlib import pyplot as plt
import twitter 
import tweepy as tw 
import csv
import pandas as pd
import numpy as np
import time
import sys
import json
import glob
from json import JSONEncoder
class MyEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__  

# In-built regular expressions library
import re 

# Set notebook mode to work in offline
import plotly.offline as pyo 
pyo.init_notebook_mode()

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set working directory
import os
os. getcwd()
os.chdir('C:/Users\Omar\Desktop\Spring 2021\Web & Social Media Analytics\Assessment\Final_code')

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

CONSUMER_KEY = creds['CONSUMER_KEY']
CONSUMER_SECRET = creds['CONSUMER_SECRET']
ACCESS_TOKEN = creds['ACCESS_TOKEN']
ACCESS_SECRET = creds['ACCESS_SECRET']

# Authorization and Authentication
auth = tw.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET) 
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)


api = tw.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True) 

# Collect tweets using the Cursor object
# Each item in the iterator has various attributes that you can access to get information about each tweet
def get_tweets(search_query, num_tweets,date_since):
    tweet_list = [tweets for tweets in tw.Cursor(api.search,
                                    q=search_query,
                                    lang="en",
                                    since=date_since,
                                    tweet_mode='extended').items(num_tweets)]
    # Begin scraping the tweets individually:
    for tweet in tweet_list:
        tweet_id = tweet.id # get Tweet ID result
        created_at = tweet.created_at # get time tweet was created
        text = tweet.full_text # retrieve full tweet text
        screen_name = tweet.user.screen_name # retrieve screen name
        verified  = tweet.user.verified # verified or not
        location = tweet.user.location # retrieve user location
        retweet = tweet.retweet_count # retrieve number of retweets
        favorite = tweet.favorite_count # retrieve number of likes
        tweet_source = tweet.source # retrieve device of tweet
        with open('tweets_AAM.csv','a', newline='', encoding='utf-8') as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=',') 
            csv_writer.writerow([tweet_id, created_at, text, screen_name, verified, location, retweet, favorite, tweet_source]) # write each row
 
#Choosing a date to start searching for tweets    
date_since = "2021-03-01"
          
# Specifying exact phrase to search for. This is not case senstitive
search_words = "#StopAsianHate OR #StopAAPIHate"

# Exclude Links, retweets, replies
search_query = search_words + " -filter:retweets AND -filter:replies"      
get_tweets(search_query,1500,date_since)          

# Data reading & pre-processing
tweets = []      
df = pd.read_csv('tweets_AAM.csv', index_col = None, header = 0) # Convert each csv to a dataframe
tweets.append(df)
tweets_df = pd.concat(tweets, axis=0, ignore_index = True) # Merge all dataframes
tweets_df.columns=['tweet_id','created_at','text','screen_name','verified','location','retweet','favorite','tweet_source']
tweets_df.head()
tweets_df.shape    
tweets_df.duplicated(subset='tweet_id').sum()
tweets_df = tweets_df.drop_duplicates('tweet_id')
tweets_df['location']=tweets_df['location'].fillna('No location')
tweets_df.isna().any()

#Get Hashtags from tweets and visualize it  
def getHashtags(tweet):
    tweet = tweet.lower()  #has to be in place
    tweet = re.findall(r'\#\w+',tweet) # Remove hastags with REGEX
    return " ".join(tweet)

tweets_df['hashtags'] = tweets_df['text'].apply(getHashtags)
tweets_df.head()
hashtags_list = tweets_df['hashtags'].tolist()

# Iterate over all hashtags so they can be split where there is more than one hashtag per row
hashtags = []
for item in hashtags_list:
    item = item.split()
    for i in item:
        hashtags.append(i)

# Use the Built-in Python Collections module to determine Unique count of all hashtags used
from collections import Counter
counts = Counter(hashtags)
hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
hashtags_df.columns = ['hashtags', 'hashtag_count']
hashtags_df.sort_values(by='hashtag_count', ascending=False, inplace=True)
print (f'Total Number of Unique Hashtags is: {hashtags_df.shape[0]}.')

hashtags_df["percentage"] = 100*(hashtags_df["hashtag_count"]/hashtags_df['hashtag_count'].sum())
hashtags_df = hashtags_df.head(10)

# Devices List
devices_list = tweets_df['tweet_source'].tolist()

# Iterate over all hashtags so they can be split where there is more than one hashtag per row
devices = []
for item in devices_list:
    #item = item.split()
    #for i in item:
    devices.append(item)

# Use the Built-in Python Collections module to determine Unique count of all hashtags used
from collections import Counter
source_counts = Counter(devices)
devices_df = pd.DataFrame.from_dict(source_counts, orient='index').reset_index()
devices_df.columns = ['devices', 'devices_counts']
devices_df.sort_values(by='devices_counts', ascending=False, inplace=True)
print (f'Total Number of Unique devices is: {devices_df.shape[0]}.')


devices_df["Percentage"] = 100*(devices_df["devices_counts"]/devices_df['devices_counts'].sum())
devices_df = devices_df.head(10)

plt.bar(devices_df["devices"], devices_df["Percentage"]) 
plt.show()

# HeatMap Creation
from geopy.geocoders import Nominatim
import gmplot

geolocator = Nominatim(user_agent="http")

# Go through all tweets and add locations to 'coordinates' dictionary
coordinates = {'latitude': [], 'longitude': []}
for count, user_loc in enumerate(tweets_df['location']):
    try:
        location = geolocator.geocode(user_loc)
        
        # If coordinates are found for location
        if location:
            coordinates['latitude'].append(location.latitude)
            coordinates['longitude'].append(location.longitude)
            
    # If too many connection requests
    except:
        pass
    
# Instantiate and center a GoogleMapPlotter object to show our map
gmap = gmplot.GoogleMapPlotter(30, 0, 3)

# Insert points on the map passing a list of latitudes and longitudes
gmap.heatmap(coordinates['latitude'], coordinates['longitude'], radius=20)

# Save the map to html file
gmap.draw("python_heatmap.html")

#Sentiment Analysis Report
#Finding sentiment analysis (+ve, -ve and neutral)
pos = 0
neg = 0
neu = 0

tweet_Sentiment=[]

sia = SentimentIntensityAnalyzer()

for index, row in tweets_df.iterrows():    
    analysis = sia.polarity_scores(row['text'])

    
    if analysis["compound"]>0:
       pos = pos +1
       tweet_Sentiment.append('Positive')
       
    elif analysis["compound"]<0:
       neg = neg + 1
       tweet_Sentiment.append('Negative')
    else:
       neu = neu + 1
       tweet_Sentiment.append('Neutral')

tweets_df['Sentiment'] = tweet_Sentiment

print("Total Positive = ", pos)
print("Total Negative = ", neg)
print("Total Neutral = ", neu)


# Plotting sentiments
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neu]
colors = ['yellowgreen','lightcoral','gold']
explode = (0.1, 0, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Adding graph interactions 
datadir = "Followers_data"

# Authorization and Authentication / twitter library 
t = twitter.Api(consumer_key = CONSUMER_KEY,
                 consumer_secret = CONSUMER_SECRET,
                 access_token_key = ACCESS_TOKEN,
                 access_token_secret = ACCESS_SECRET,sleep_on_rate_limit=True)

tweets_df.sort_values(by='retweet', ascending=False, inplace=True)
tweets_df_top = tweets_df[:50]
Top_retweet_users = list(tweets_df_top['screen_name'].unique())
count = 0  
nameCount = len(Top_retweet_users)
 
for sn in Top_retweet_users:
         """
         For each user, get the followers and tweets and save them
         to output pickle and JSON files.
         """
         timeline = t.GetUserTimeline(screen_name=sn, count=100)
         tweets = [i.AsDict() for i in timeline]
         with open(datadir + "/" + sn + ".tweets.json", "w") as tweetsjson:
             json.dump(tweets, tweetsjson) # Store the informtion in a JSON. 
         time.sleep(70) # avoids hitting Twitter rate limit
         # Progress bar to track approximate progress
         count +=1
         per = round(count*100.0/nameCount,1)
         sys.stdout.write("\rTwitter call %s%% complete." % per)
         sys.stdout.flush()    


path_to_json = os. getcwd() + "\\" + datadir
json_pattern = os.path.join(path_to_json,'*.tweets.json')
file_list = glob.glob(json_pattern)
temp= pd.DataFrame()
df= pd.DataFrame()

for file in file_list:
    temp = pd.read_json(file)
    if df.empty:
       df = temp
    else:
       df = df.append(temp)
    
df=df.reset_index(drop=True)    

def getbasics(tfinal):
     """
     Get the basic information about the user.
     """
     tfinal["screen_name"] = df["user"].apply(lambda x: x["screen_name"])
     tfinal["user_id"] = df["user"].apply(lambda x: x["id"])
     #tfinal["followers_count"] = df["user"].apply(lambda x: x["followers_count"])
        
     return tfinal
 
def getretweets(tfinal):
     """
     Get retweets.
     """
     # Inside the tag "retweeted_status" will find "user" and will get "screen name" and "id". 
     tfinal["retweeted_screen_name"] = df["retweeted_status"].apply(lambda x: x["user"]["screen_name"] if x is not np.nan else np.nan)
     tfinal["retweeted_id"] = df["retweeted_status"].apply(lambda x: x["user"]["id_str"] if x is not np.nan else np.nan)
     return tfinal
 
def getinreply(tfinal):
     """
     Get reply info.
     """
     # Just copy the "in_reply" columns to the new DataFrame.
     if 'in_reply_to_screen_name' in df.columns:
         tfinal["in_reply_to_screen_name"] = df["in_reply_to_screen_name"]
     if 'in_reply_to_status_id' in df.columns:    
         tfinal["in_reply_to_status_id"] = df["in_reply_to_status_id"]
     if 'in_reply_to_user_id' in df.columns:    
         tfinal["in_reply_to_user_id"]= df["in_reply_to_user_id"]
     return tfinal

def filldf(tfinal):
     """
     Put it all together.
     """
     getbasics(tfinal)
     getretweets(tfinal)
     getinreply(tfinal)
     return tfinal  

def getinteractions(row):
     """
     Get the interactions between different users.
     """
     # From every row of the original DataFrame.
     # First obtain the "user_id" and "screen_name".
     user = row["user_id"], row["screen_name"]
     # Be careful if there is no user id.
     if user[0] is None:
         return (None, None), []           
     # The interactions are going to be a set of tuples.
     interactions = set()

     # Add all interactions. 
     # First, add the interactions corresponding to replies adding 
     # the id and screen_name.
     interactions.add((row["in_reply_to_user_id"], 
     row["in_reply_to_screen_name"]))
     # After that, we add the interactions with retweets.
     interactions.add((row["retweeted_id"], 
     row["retweeted_screen_name"]))
     # And later, the interactions with user mentions.
     interactions.add((row["user_mentions_id"], 
     row["user_mentions_screen_name"]))

     # Discard if user id is in interactions.
     interactions.discard((row["user_id"], row["screen_name"]))
     # Discard all not existing values.
     interactions.discard((None, None))
     # Return user and interactions.
     return user, interactions        

tfinal = pd.DataFrame(columns = ["created_at", "id", "in_reply_to_screen_name",
                                       "in_reply_to_status_id", "in_reply_to_user_id",
                                        "retweeted_id", "retweeted_screen_name",
                                        "user_mentions_screen_name",
                                        "user_mentions_id", "text", "user_id", "screen_name",
                                        "followers_count"])         
         
eqcol = ["created_at", "id", "text"]
tfinal[eqcol] = df[eqcol]
tfinal = filldf(tfinal)
tfinal = tfinal.where((pd.notnull(tfinal)), None)

# Graph Analysis     
graph = nx.Graph()

for index, tweet in tfinal.iterrows():
     user, interactions = getinteractions(tweet)
     user_id, user_name = user
     tweet_id = tweet["id"]
     for interaction in interactions:
         int_id, int_name = interaction
         graph.add_edge(user_name, int_name, tweet_id=tweet_id)
         graph.nodes[user_name]["name"] = user_name
         graph.nodes[int_name]["name"] = int_name 
         
degrees = [val for (node, val) in graph.degree()]
print("The maximum degree of the graph is " + str(np.max(degrees))) 
print("The minimum degree of the graph is " + str(np.min(degrees)))
print("There are " + str(graph.number_of_nodes()) + " nodes and " + str(graph.number_of_edges()) 
      + " edges present in the graph")
print("The average degree of the nodes in the graph is " + str(np.mean(degrees)))
         
if nx.is_connected(graph):
     print("The graph is connected")
else:
     print("The graph is not connected")
print("There are " + str(nx.number_connected_components(graph)) + " connected in the graph.")

# Finding the largest subgraph         
largestsubgraph = max([graph.subgraph(c) for c in nx.connected_components(graph)], key=len)
print("There are " + str(largestsubgraph.number_of_nodes()) 
       + " nodes and " + str(largestsubgraph.number_of_edges()) 
       + " edges present in the largest component of the graph.")

print("The average clustering coefficient is " + str(nx.average_clustering(largestsubgraph)) 
      + " in the largest subgraph")
print("The transitivity of the largest subgraph is " + str(nx.transitivity(largestsubgraph)))
print("The diameter of our graph is " + str(nx.diameter(largestsubgraph)))
print("The average distance between any two nodes is " 
      + str(nx.average_shortest_path_length(largestsubgraph)))         

#Plot the graph
G_fb = graph
pos = nx.spring_layout(G_fb)
betCent =nx.betweenness_centrality(G_fb, normalized =True, endpoints=True)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb, pos=pos, with_labels = False, node_color=node_color, node_size=node_size)
plt.axis('off')
sorted(betCent, key=betCent.get, reverse=True)[:5]

#Plot the largest subgraph
G_fb = largestsubgraph
pos = nx.spring_layout(G_fb)
betCent =nx.betweenness_centrality(G_fb, normalized =True, endpoints=True)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb, pos=pos, with_labels = False, node_color=node_color, node_size=node_size)
plt.axis('off')
sorted(betCent, key=betCent.get, reverse=True)[:5]

# Community Detection
TwitterNet=largestsubgraph

# Calling the louvain algorithm to detect communities 
coms = algorithms.louvain(TwitterNet, weight="weight", resolution=1)

# Giving the proper layout for the visualisation.
pos = nx.spring_layout(TwitterNet)

# Using the viz module to visualise the clusters and graph based on the coms
viz.plot_network_clusters(TwitterNet,coms,pos)
viz.plot_community_graph(TwitterNet,coms)

# The number of the communities detected
len(coms.communities)
    
# Print all the communities with their nodes names in a nested list
coms.communities

# Centrality Measures
g=TwitterNet
pos = nx.spring_layout(g)

# Calculating Centrality metrics for the Graph
dict_degree_centrality = nx.degree_centrality(g)
df_degree_centrality = pd.DataFrame.from_dict({
    'node': list(dict_degree_centrality.keys()),
    'centrality': list(dict_degree_centrality.values())
})
df_degree_centrality = df_degree_centrality.sort_values('centrality', ascending=False)

# Calculating Closeness Centrality
dict_closeness_centrality = nx.closeness_centrality(g)
df_closeness_centrality = pd.DataFrame.from_dict({
    'node': list(dict_closeness_centrality.keys()),
    'centrality': list(dict_closeness_centrality.values())
})
df_closeness_centrality = df_closeness_centrality.sort_values('centrality', ascending=False)

# Calculating Betweeness Centrality
dict_betweenness_centrality = nx.betweenness_centrality(g)
df_betweenness_centrality = pd.DataFrame.from_dict({
    'node': list(dict_betweenness_centrality.keys()),
    'centrality': list(dict_betweenness_centrality.values())
})
df_betweenness_centrality = df_betweenness_centrality.sort_values('centrality', ascending=False)

