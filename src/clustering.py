############################################################################
# Perform main clusering and pruning
# Based on <location> paramenter.
############################################################################


#!/usr/bin/env python
# coding: utf-8


from .misc_lib import *
import re, os, tqdm, glob, sys
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams
from pyvis import network as pvnet


#####################################################################
# main options:
# location = 'ontario'
# location = 'british_columbia'
# location = 'quebec'
location = sys.argv[1]


# Some part take a long time to run. To reduce rerunning these portions,
# output is sved and can be read in instead of regernreating it. 
# These flags control whether regeneration should run.
GEN_CENTR = True
GEN_MAINTWEETS = True
RERUN_SUBFOLLOW = True
GEN_COMM = True

# Keep top 50 author nodes
top_N = 50

def clamp(rgb): 
    r,g,b = [max(0, min(round(255*x), 255)) for x in rgb]
    return "#{0:02x}{1:02x}{2:02x}".format(r,g,b)

def plot_g_pyviz(G, name='out.html', height='500px', width='500px', graph_type='atlas'):
    g = G.copy() # some attributes added to nodes
    net = pvnet.Network(notebook=True, directed=True, height=height, width=width)
    net.inherit_edge_colors(True)
    if graph_type == 'atlas':
        opts = '''
            var options = {
              "physics": {
              "enabled": true,
                "forceAtlas2Based": {
                  "gravitationalConstant": -100,
                  "centralGravity": 0.11,
                  "springLength": 100,
                  "springConstant": 0.09,
                  "avoidOverlap": 1
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.22
              }
            }
        '''
    else:
        opts = '''
             var options = {
              "physics": {
                "enabled": true}
            }
        '''

    net.set_options(opts)
    net.from_nx(g)

    return net.show(name)





filename = './data/%s_retweets.csv'%(location)
df = pd.read_csv(filename)
tweets = df[['tweeter_username','username','tweet_id']]
for c in ['tweeter_username','username']:
    tweets[c] = tweets[c].apply(lambda u: u.lower())
# tweets = tweets.values.tolist()
if GEN_CENTR:
    edges = tweets[['tweeter_username','username','tweet_id']].copy()
    edges.columns = ['from','to','N']
    weighted = edges.groupby(['from','to'])['N'].count().reset_index(drop=False)
    weighted['weight'] = 1/weighted['N']

    G = nx.DiGraph()
    _=[G.add_edge(x,y,weight=w, title=n, size=n) for [x,y,w,n] in weighted[['to','from','weight','N']].values]
    mapping = list(G.nodes())
    adj = nx.adjacency_matrix(G).todense()
    adj = np.array(adj)
    print(adj.shape)
    for n in G.nodes():
        G.nodes[n]['size'] = (np.log1p(G.degree(n))+1)**1.7

    prank_df = pd.DataFrame([nx.pagerank(G, alpha=0.9)]).T.reset_index(drop=False)
    prank_df.columns = ['node','prank']
    prank_df = prank_df.sort_values(by=['prank'])
    btwn_df = pd.DataFrame([nx.betweenness_centrality(G)]).T.reset_index(drop=False)
    btwn_df.columns = ['node','btwn']
    btwn_df = btwn_df.sort_values(by=['btwn'])
    centr_df = pd.DataFrame([nx.eigenvector_centrality(G)]).T.reset_index(drop=False)
    centr_df.columns = ['node','centr']
    centr_df.sort_values(by=['centr'])

    centralities = prank_df.merge(btwn_df, on=['node']).merge(centr_df, on=['node'])
    centralities.to_csv('./data/results/centralities_%s.csv'%(location),index=False)
    centralities = pd.read_csv('./data/results/centralities_%s.csv'%(location))
else:
    centralities = pd.read_csv('./data/results/centralities_%s.csv'%(location))


btwn_th = centralities.quantile(.8).btwn
print(btwn_th)
btwn_top = centralities[centralities.btwn>btwn_th]


prank_th = centralities.quantile(.8).prank
print(prank_th)
prank_top = centralities[centralities.prank>prank_th]

centr_th = centralities.quantile(.8).centr
print(centr_th)
centr_top = centralities[centralities.centr>centr_th]

print(centralities.columns)
df = btwn_top[['node','btwn']].merge(prank_top[['node','prank']], on=['node'],how='outer').\
    merge(centr_top[['node','centr']], on=['node'],how='outer').\
    dropna(subset=['node']).\
    fillna(0.0)

#############################################
# top nodes
for c in ['centr','prank','btwn']:
    df[c+'n'] = df[c]/df[c].max()
df['aggr1'] = df[['centrn','prankn']].mean(axis=1)
df['aggr2'] = df[['centrn','prankn','btwnn']].mean(axis=1)
idx1 = df.sort_values(by=['aggr1','centrn','prankn','btwnn'], ascending=[False,False,False,False]).head(top_N).index
idx2 = df[df.btwnn>0].sort_values(by=['btwnn','aggr2','centrn','prankn'], ascending=[False,False,False,False]).head(top_N).index
df.loc[idx1,'top1'] = 1
df.loc[idx2,'top2'] = 1

profiles = df.loc[idx1.append(idx2).unique()]


##############################################################
# get top users
top1_users = profiles[(profiles.top1==1)]
top2_users = profiles[(profiles.top1!=1)&(profiles.top2==1)]
sub_tweets = tweets[(tweets.tweeter_username.isin(top1_users.node))&\
                    (tweets.username.isin(top2_users.node))]
print(top1_users.node)
print(top2_users.node)
print(sub_tweets.shape[0], sub_tweets.tweet_id.nunique())


##################################################
# get followers
import glob, re
follower_files = glob.glob('./data/degree_progress/*.csv')
res = []
for f in tqdm.tqdm(follower_files, total=len(follower_files), desc='load follows'):
    username = re.match(r'.*/(.+)_followers\.csv',f).groups()[0]
    if username in profiles.node.values:
        tmp = pd.read_csv(f)
        tmp['node'] = username.lower()
        tmp['username'] = tmp['username'].apply(lambda u: u.lower())
        res.append(tmp[['node','username']])

follow_df = pd.concat(res).drop_duplicates()
print(follow_df.node.nunique())

##########################################################
# get main tweets to focus on 
if GEN_MAINTWEETS:
    main_tweets = []
    for tweet_id,data_sub in tqdm.tqdm(tweets[tweets['tweet_id'].isin(sub_tweets.tweet_id)].groupby(['tweet_id']), total=sub_tweets.tweet_id.nunique(), desc='main tweets'):
        tweeter = data_sub.iloc[0].tweeter_username.lower()
        names = list(set(list(data_sub.tweeter_username) + list(data_sub.username)))
        names.sort()
        follow_sub = [[f,u]  for f,u in follow_df.values if f in names and u in names ]
        
        c21,c31 = len(follow_sub), len(names)
        if (c21*c31)>0:
            main_tweets.append([tweet_id, tweeter, c21,c31])
            #print(tweet_id, tweeter, c21,c31)
    main_tweets = pd.DataFrame(main_tweets, columns = ['tweet_id', 'tweeter', 'follows', 'names'])


    #############################################
    # Sort top tweets
    main_tweets = main_tweets.merge(centralities, left_on=['tweeter'], right_on=['node']).\
            sort_values(by=['prank','follows','names'], ascending=False)
    for col in ['node','username','followers_count']:
        try:
            main_tweets = main_tweets.drop(col, axis=1)
        except KeyError:
            pass
    main_tweets.to_csv('./data/graphs/%s_main_tweets_sub_follower.csv'%(location))
else:
    main_tweets = pd.read_csv('./data/graphs/%s_main_tweets_sub_follower.csv'%(location))


##########################################
# gen follow usernames

if RERUN_SUBFOLLOW:
    names2 = []
    follow_sub2 = []
    for tweet_id in tqdm.tqdm(main_tweets.tweet_id, desc='subflow'):
        data_sub = tweets[tweets.tweet_id==tweet_id]
        tweeter = data_sub.tweeter_username.iloc[0]
        names1 = list(set(list(data_sub.tweeter_username) + list(data_sub.username)))
        names2 += names1
        follow_sub1 = [[f,u]  for f,u in follow_df.values if f in names and u in names1 ]
        follow_sub2 += follow_sub1
        pd.DataFrame(names2).to_csv('./data/graphs/%s_names2_sub_followers.csv'%(location), index=False)
        pd.DataFrame(follow_sub2).to_csv('./data/graphs/%s_follow_sub2_sub_followers.csv'%(location), index=False)
else:
    names2 = pd.read_csv('./data/graphs/%s_names2_sub_followers.csv'%(location)).values.tolist()
    names2 = flatten(names2)
    follow_sub2 = pd.read_csv('./data/graphs/%s_follow_sub2_sub_followers.csv'%(location)).values.tolist()

#########################################
# Clean up follows and list of names
follow_sub = dedup(follow_sub2)
follow_sub = [[f1.lower(),f2.lower()] for f1,f2 in follow_sub]
names = dedup(names2)
names.sort()
names = [n.lower() for n in names]
names_i = dict(zip(names, range(len(names))))


##############################################
# load edges into graph
G2 = nx.MultiDiGraph()
_=[G2.add_node(n) for n in names]
_=[G2.add_edge(x,y, color='#7AD7F0') for [x,y] in follow_sub]
G2_draw = G2.copy()
data = tweets.values
color_tab = []
import colorsys
import random
random.seed(42)
for i in range(main_tweets.shape[0]):
    r,g,b = random.randint(200,255),random.randint(100,255),random.randint(100,255)
    rgb = colorsys.hsv_to_rgb(r/255,g/255,b/255)
    color_tab.append(rgb)
    print(i, [round(255*x) for x in rgb])
for i,tweet_id in enumerate(tqdm.tqdm(main_tweets.tweet_id, desc='G2_draw')):
    _=[G2_draw.add_edge(x,y, title=str(d)) for [x,y,d] in data if d==tweet_id]
    _=[G2_draw.add_edge(x,y, color=clamp(color_tab[i]), title=str(d)) for [x,y,d] in data if d==tweet_id]

main_tweets.groupby('tweeter').max().drop(columns=['tweet_id','follows','names']).\
    merge(main_tweets.groupby('tweeter').mean().drop(columns=['tweet_id', 'prank','btwn','centr']), left_index=True, right_index=True).rename(columns={'follows':'follows_mean', 'names':'names_mean'}).\
    merge(main_tweets.groupby('tweeter').std().drop(columns=['tweet_id', 'prank','btwn','centr']), left_index=True, right_index=True).rename(columns={'follows':'follows_mean', 'names':'names_mean'}).\
    merge(pd.DataFrame(main_tweets.tweeter.value_counts()), left_index=True, right_index=True).\
    sort_values(by=['tweeter'],ascending=False).\
    to_csv('./data/graphs/sub_tweets_top_user_stats_%s.csv'%(location),index=True)



###############################################
# Create clusters
###############################################
g = G2_draw.copy()
top_adj = nx.adjacency_matrix(g).todense()
top_adj = np.array(top_adj)

print(top_adj.shape)


import networkx as nx

if GEN_COMM:
    from communities.algorithms import girvan_newman, hierarchical_clustering, louvain_method, bron_kerbosch
    communities,_ = girvan_newman(top_adj)

    def flatten1(l):
        return list(itert.chain.from_iterable(l))

    tmp = [[(i,n) for n in c] for i,c in enumerate(communities)]
    communities_i_df = pd.DataFrame(flatten(tmp), columns=['group', 'node_i'])

    # select nodes with >1 degree
    keep_nodes_i = []
    for c in communities:
        if len(c)>1:
            keep_nodes_i.append(c)
    print('org nodes', len(communities)-1) # subtract 1 for the entire graph cluster
    print('kept nodes', len(keep_nodes_i)-1) # subtract 1 for the entire graph cluster

    keep_nodes_i = flatten1(keep_nodes_i)
    keep_nodes_df = pd.DataFrame(
        [(int(i),n) for i,n in enumerate(g.nodes) if i in keep_nodes_i],
        columns=['node_i', 'node_username']
    )
    keep_nodes = np.array(keep_nodes_df.values)
    community_nodes_df = keep_nodes_df.merge(communities_i_df, on='node_i', how='inner')
    print(community_nodes_df.shape, keep_nodes_df.shape, communities_i_df.shape)
    assert community_nodes_df.shape[0] == keep_nodes_df.shape[0]
    assert keep_nodes_df.shape[0] == communities_i_df.shape[0]
    community_nodes_df.to_csv('./data/graphs/%s_communities_tweets_sub_follower.csv'%(location))
else:
    community_nodes_df = pd.read_csv('./data/graphs/%s_communities_tweets_sub_follower.csv'%(location))
    communities = []

    for i,grp in community_nodes_df.groupby('group'):
        communities.append(set(grp.node_i.values))
        # select nodes with >1 degree
        keep_nodes_i = []
        for c in communities:
            if len(c)>1:
                keep_nodes_i.append(c)
        keep_nodes_i = flatten1(keep_nodes_i)
        keep_nodes_df = pd.DataFrame(
            [(int(i),n) for i,n in enumerate(g.nodes) if i in keep_nodes_i],
            columns=['node_i', 'node_username']
        )
        keep_nodes = np.array(keep_nodes_df.values)



##########################################################
# greate subgraph with just kept nodes
##########################################################
map3 = {}
for i,k in keep_nodes:
    map3[k] = [int(i)]
g2 = g.subgraph(keep_nodes[:,1])
keep_nodes2 = np.array([(i,n) for i,n in enumerate(g2.nodes) if i in keep_nodes_i])
for i,k in keep_nodes2:
    map3[k].append(int(i))
  
# assign clusters to subgraph g2
mapping = list(g.nodes())
print(communities)
# reset all clusters
for n in tqdm.tqdm(g2.nodes(), desc='groups 1'):
    g2.nodes[n]['group'] = -1
# assign clusters
for i,c in tqdm.tqdm(enumerate(communities), total=len(communities), desc='communities'):
    if len(c) > 1:
        print((i,len(c)), end='')
    for ni in c:
        if len(c) > 1:
            ni2 = map3[mapping[ni]][0]
            g2.nodes[mapping[ni2]]['group'] = i

g2_2 = g2.copy()
for n in tqdm.tqdm(g2.nodes(), desc='groups 2'):
    g2_2.nodes[n]['group'] = -1

for group, grp in community_nodes_df.groupby(['group']):
    if grp.shape[0] > 1:
        for ni in grp['node_username']:
            g2_2.nodes[ni]['group'] = group


############################################
# generate graph
############################################
plot_g_pyviz(g, name='./data/graphs/girv_main2_%s_other_sub_follower.html'%(location), width='800px', height='800px', graph_type='other') #top + girvan_newman

#############################################
# save graph with community groups
############################################
df_g = pd.DataFrame([[user, dt['group']] for user,dt in  list(g.nodes(data=True))])
df_g.to_csv('./data/graphs/node_groups_%s_2.csv'%(location),index=False)



