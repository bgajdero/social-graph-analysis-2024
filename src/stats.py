#!/usr/bin/env python
# coding: utf-8


from misc_lib import *
import re, os, tqdm, glob
import pandas as pd, numpy as np


# data for Healthcre topic twets
DATA_PATH_IN = '../data/2023/output_tweets_cleaned_jan-March_sentiments_themed.csv'


###################################################
# COVID-19 Topic
###################################################

stats_df = []
for location  in ['quebec', 'ontario', 'british_columbia']:
    stats_s = {}
    filename = '../data/%s_retweets.csv'%(location)
    df = pd.read_csv(filename)
    tweets = df[['tweeter_username','username','tweet_id']]
    for c in ['tweeter_username','username']:
        tweets[c] = tweets[c].apply(lambda u: u.lower())
    centralities = pd.read_csv('../results/centralities_%s.csv'%(location))

    org_authors_n = tweets.tweeter_username.nunique()
    org_sharer_n = tweets.username.nunique()
    shares_n = tweets.tweet_id.shape[0]
    tweets_n = tweets.tweet_id.nunique()
    profiles_n = centralities.shape[0]

    # clusters
    try:
        df_g = pd.read_csv('../graphs/node_groups_%s_2.csv'%(location))
    except FileNotFoundError:
        df_g = pd.read_csv('../graphs/node_groups_%s.csv'%(location))
    df_g.columns = ['username','group']



    bias = pd.read_csv('graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    bias = bias[~pd.isnull(bias.prank)]

    cluster_n = sharer_n = df_g['group'].nunique()
    author_n = bias.shape[0]
    sharer_n = df_g['group'].value_counts().sum()
    sharer_n = sharer_n - author_n

    stats_s = pd.Series({
        'location':location,
        'tweets_n':tweets_n,
        'profiles_n':profiles_n,
        'shares_n':shares_n,
        'cluster_n':cluster_n,
        'author_n': author_n,
        'sharer_n':sharer_n,
        'org_authors_n': org_authors_n,
        'org_sharer_n':org_sharer_n,
    })
    stats_df += [stats_s]

stats_df = pd.DataFrame(stats_df)
# display results
stats_df
stats_df.sum()



###################################################
# Healthcare topic
###################################################
def get_healthcare_tweets(location):
    print(location)
    # from manual file for Core Users
    bias = pd.read_csv('../graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    bias = bias[~pd.isnull(bias.prank)]

    import json
    core_user_ids = pd.DataFrame(json.loads(open('../data/2023/all_user_ids.json').read()).items(), columns=['tweeter_username','author_id'])
    core_user_ids.author_id = core_user_ids.author_id.astype(int)

    # Jan 1 - March 15, 2023
    output_tweets_cleaned_Jan_March  = pd.read_csv(DATA_PATH_IN)\
        .rename(columns={'id':'tweet_id'})\
        .drop_duplicates(['tweet_id'])
    output_tweets_cleaned_Jan_March['author_id'] = output_tweets_cleaned_Jan_March['author_id'].astype(int)
    output_tweets_cleaned_Jan_March = output_tweets_cleaned_Jan_March\
        .merge(core_user_ids, on='author_id', how='inner')\
        .drop('author_id', axis=1)
    

    # Dec 2022 - March 2023
    retweeters_fixed = pd.read_csv('../data/2023/retweeters_fixed.csv')\
        .drop('retweeter_id',axis=1)\
        .rename(columns={'retweeter_username':'username'})
    retweeters_fixed['action'] = 'retweet'
    # Dec 2022 - March 2023
    output_retweets_Nov =  pd.read_csv('../data/2023/output_retweets_Nov.csv')\
        .drop('retweeter_id',axis=1)\
        .rename(columns={'retweeter_username':'username'})
    output_retweets_Nov['action'] = 'retweet'
    # Dec 31, 2022 - March 15, 2023
    likers_fixed = pd.read_csv('../data/2023/likers_fixed.csv')\
        .drop('liker_id',axis=1)\
        .rename(columns={'liker_username':'username'})
    likers_fixed['action'] = 'like'

    retweeters_combined = pd.concat([output_retweets_Nov,retweeters_fixed, likers_fixed])\
        .drop_duplicates()\
        .merge(output_tweets_cleaned_Jan_March, on='tweet_id', how='right')
    tweets = retweeters_combined[['tweeter_username','username','tweet_id']]
    tweets = tweets[tweets.tweeter_username.isin(bias.tweeter_username)]



    tweets['tweeter_username'] = tweets.tweeter_username.apply(lambda u: u.lower() if type(u)==str else u)
    tweets['username'] = tweets.username.apply(lambda u: u.lower() if type(u)==str else u)

    # tweets after truncation in cluster_truncate
    main_tweets = output_tweets_cleaned_Jan_March[['tweeter_username', 'tweet_id']]\
        .copy()\
        .rename(columns={'tweeter_username':'tweeter'})
    main_tweets = main_tweets[main_tweets.tweeter.isin(bias.tweeter_username)]

    return tweets, main_tweets, bias

stats2_df = []
# counts
for location  in ['quebec', 'ontario', 'british_columbia']:
    stats_s = {}
    tweets, main_tweets, bias = get_healthcare_tweets(location=location)

    shares_n = tweets.tweet_id.shape[0]
    sharer_n = tweets.username.nunique()
    tweets_n = tweets.tweet_id.nunique()
    profiles_n = len(list(set(list(tweets.username.unique()) + list(tweets.tweeter_username.unique()))))
    author_n = bias.shape[0]
    sharer_n = sharer_n - author_n

    stats_s = pd.Series({
        'location':location,
        'tweets_n':tweets_n,
        'profiles_n':profiles_n,
        'shares_n':shares_n,
        'author_n': author_n,
        'sharer_n':sharer_n
    })
    stats2_df += [stats_s]

stats2_df = pd.DataFrame(stats2_df)
stats2_df
stats2_df.sum()

df_g = []
for location  in ['quebec', 'ontario', 'british_columbia']:
    _tweets = pd.read_csv('graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    _tweets = _tweets[~pd.isnull(_tweets.prank)]
    _tweets['location'] = location
    _bias = pd.read_csv('graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    _bias = _bias[~pd.isnull(_bias.prank)]



pd.concat([stats_df, stats2_df]).sum()