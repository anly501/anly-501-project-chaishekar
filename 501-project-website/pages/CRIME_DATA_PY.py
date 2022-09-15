import pandas as pd
import os
import time
import requests
import json
import csv
from tqdm import tqdm
import tweepy
import requests
import pandas as pd
import os
consumer_key        = 'LBNeNod2Bd3Y0AdGBJT8Ka4Aa'
consumer_secret     = 'MNlkJwxEeqS0MXkpi2wNidrAbiUpf4ueVkBJcvZsBBtvne1A7F'
access_token        = '1568064617036881922-LI9dm4r6ISllYfbKE4m9Vz2OW0MZI5'
access_token_secret = 'MCq785e0hNBUkkkSwmQFHglVs6hMMKFh1pylkpCCVwyGa'
bearer_token        = 'AAAAAAAAAAAAAAAAAAAAAIKDgwEAAAAAQNemNVFuPsQSzPbcM%2FvvMYx6EyI%3DtKOnXddTRlvH45AkNclsSwlEn8iBC4sEBoUDaAjqJU0mZKDNzm'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
headers = {"Authorization": "Bearer {}".format(bearer_token)}
print()
def search_twitter (query, tweet_fields, max_results, bearer_token = bearer_token) :
    headers = {"Authorization": "Bearer {}".format (bearer_token)}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {"query": query, "tweet.fields": tweet_fields, "max_results": max_results}
    response = requests. request ("GET", url, headers=headers, params = params)
    if response.status_code != 200:
        raise Exception (response.status_code, response.text)
    return response. json()
query = "crime location usa"
max_results = 100
tweet_fields="text,author_id,created_at,lang"
# twitter api call
data= search_twitter(query, tweet_fields, max_results, bearer_token = bearer_token)
print(json.dumps(data, indent=4, sort_keys=True))

