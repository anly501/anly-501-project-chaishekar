import numpy as np
import tweepy
import tweepy as tw
import pandas as pd
# your Twitter API key and API secret
api_key = "LBNeNod2Bd3Y0AdGBJT8Ka4Aa"
api_secret = "MNlkJwxEeqS0MXkpi2wNidrAbiUpf4ueVkBJcvZsBBtvne1A7F"
# authenticate
auth = tw.OAuthHandler(api_key, api_secret)
api = tw.API(auth, wait_on_rate_limit=True)
search_query = "#crime -filter:retweets"
# get tweets from the API
tweets = tw.Cursor(api.search_tweets,
              q=search_query,
              lang="en").items(100)
# store the API responses in a list
tweets_copy = []
for tweet in tweets:
    tweets_copy.append(tweet)
    
print("Total Tweets fetched:", len(tweets_copy))
# intialize the dataframe
tweets_df = pd.DataFrame()
# populate the dataframe
for tweet in tweets_copy:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               'source': tweet.source}))
    tweets_df = tweets_df.reset_index(drop=True)
# show the dataframe
tweets_df.head()