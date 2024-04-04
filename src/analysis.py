from misc_lib import *
import re, os, tqdm, glob, random, glob, re, sys
import pandas as pd, numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


GEN_GRAPHS = True
LOCATIONS = ['quebec', 'british_columbia', 'ontario']

# When Running COVID-19
TOPIC = 'COVID'
DATA_PATH_IN = '../data/sentiment/sentiment*.csv'

# when running Heathcare
# TOPIC = 'Healhtcare'
# DATA_PATH_IN = '../data/2023/output_tweets_cleaned_jan-March_sentiments_themed.csv'


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'Liberal'/'Conservative'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def mean3d(a,b,c):
    try:
        total = sum([a,b,c])
        aa = a/total
        bb = b/total
        cc = c/total
    except ZeroDivisionError:
        total = 0.0
        aa = 0.0
        bb = 0.0
        cc = 0.0
    x = [0,cc-aa]
    y = [0,bb]
    
    # Calculating dot product using dot()
    dot = np.dot(y, x) *bb + x[1]
    dot = (dot + 1)/2
    return dot


def get_bias(location):
    print(location)
    # from manual file for Core Users
    bias = pd.read_csv('graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    bias = bias[~pd.isnull(bias.prank)]
    


    # from data
    filename = '../data/%s_retweets.csv'%(location)
    tweets = pd.read_csv(filename)
    tweets = tweets[['tweeter_username','username','tweet_id']]
    # centralities = pd.read_csv('results/centralities_%s.csv'%(location))
    # centralities['followee_username'] = centralities.node.apply(lambda u: u.lower() if type(u)==str else u)
    tweets['tweeter_username'] = tweets.tweeter_username.apply(lambda u: u.lower() if type(u)==str else u)
    tweets['username'] = tweets.username.apply(lambda u: u.lower() if type(u)==str else u)

    # tweeta after truncation in cluster_truncate
    # if location == 'quebec':
    #     main_tweets = pd.read_csv('graphs/%s_2_main_tweets_sub_follower.csv'%(location))
    # else:
    main_tweets = pd.read_csv('graphs/%s_main_tweets_sub_follower.csv'%(location))
    # tweets = tweets[(tweets.tweet_id.isin(main_tweets.tweet_id))&(tweets.username.isin(node_groups.username))]
    # main_tweets['ratio'].hist(bins=10)
    main_tweets.drop_duplicates(['tweeter'])[['tweeter','prank','btwn','centr']].sort_values(by='prank',ascending=False)

    # tmp = node_groups.merge(main_tweets, left_on='username', right_on='tweeter')
    res = []
    for (u,grp_bias),grp in main_tweets.merge(bias[['tweeter_username','bias']], left_on='tweeter', right_on='tweeter_username').groupby(['tweeter','bias']):
        # grp_bias
        tmp_u = grp.bias.value_counts()
        tmp_u['tweeter_username'] = u
        tmp_u['bias'] = grp_bias
        res.append(tmp_u)


    res = pd.DataFrame(res)
    for c in ['Liberal','Moderate','Conservative']:
        if c not in res.columns:
            res[c] = np.nan
    res = res[['tweeter_username','bias','Liberal','Moderate','Conservative']].reset_index(drop=True)
    res = res.fillna(0.0)

    res['trend'] = res.apply(lambda row: mean3d(row['Conservative'], row['Moderate'], row['Liberal']), axis=1)

    for idx, row in res.iterrows():
        if row['trend'] > 0.5:
            res.at[idx,'affiliation'] = 'Liberal'
        elif row['trend'] < 0.5:
            res.at[idx,'affiliation'] = 'Conservative'
        else:
            res.at[idx,'affiliation'] = 'Moderate'

    affiliation_tweets = res.merge(tweets, on='tweeter_username',how='left')

    affiliation_retweets = affiliation_tweets.groupby(['username','affiliation']).count().\
        bias.reset_index(drop=False).\
        pivot(index='username',columns='affiliation', values='bias').reset_index(drop=False)
    
    for c in ['Liberal','Moderate','Conservative']:
        if c not in affiliation_retweets.columns:
            affiliation_retweets[c] = 0.0
    affiliation_retweets.fillna(0.0,inplace=True)
    affiliation_retweets['trend'] = affiliation_retweets.apply(lambda row: mean3d(row['Conservative'], row['Moderate'], row['Liberal']), axis=1)
    jumped = affiliation_tweets[(affiliation_tweets.username.isin(affiliation_retweets.username))&(affiliation_tweets.tweet_id.isin(main_tweets.tweet_id))]

    jumped = jumped.merge(affiliation_retweets,on='username', suffixes=['','_username'])
    return jumped


def get_bias2(location):
    print(location)
    # from manual file for Core Users
    bias = pd.read_csv('graphs/sub_tweets_top_user_stats_%s_bias.csv'%(location), encoding = "iso-8859-1").rename(columns={'Unnamed: 0':'tweeter_username'})
    bias = bias[~pd.isnull(bias.prank)]


    # node_groups = pd.read_csv('graphs/node_groups_%s.csv'%(location))
    # node_groups.columns=['username','group']


    # edges = pd.read_csv('graphs/graph_groups_one_%s.csv'%(location))
    # from data

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

    tweets['tweeter_username'] = tweets.tweeter_username.apply(lambda u: u.lower() if type(u)==str else u)
    tweets['username'] = tweets.username.apply(lambda u: u.lower() if type(u)==str else u)

    # tweets after truncation in cluster_truncate
    main_tweets = output_tweets_cleaned_Jan_March[['tweeter_username', 'tweet_id']]\
        .copy()\
        .rename(columns={'tweeter_username':'tweeter'})

    res = []
    for (u,grp_bias),grp in main_tweets.merge(bias[['tweeter_username','bias']], left_on='tweeter', right_on='tweeter_username').groupby(['tweeter','bias']):
        # grp_bias
        tmp_u = grp.bias.value_counts()
        tmp_u['tweeter_username'] = u
        tmp_u['bias'] = grp_bias
        res.append(tmp_u)


    res = pd.DataFrame(res)
    for c in ['Liberal','Moderate','Conservative']:
        if c not in res.columns:
            res[c] = np.nan
    res = res[['tweeter_username','bias','Liberal','Moderate','Conservative']].reset_index(drop=True)
    res = res.fillna(0.0)

    res['trend'] = res.apply(lambda row: mean3d(row['Conservative'], row['Moderate'], row['Liberal']), axis=1)

    for idx, row in res.iterrows():
        if row['trend'] > 0.5:
            res.at[idx,'affiliation'] = 'Liberal'
        elif row['trend'] < 0.5:
            res.at[idx,'affiliation'] = 'Conservative'
        else:
            res.at[idx,'affiliation'] = 'Moderate'

    affiliation_tweets = res.merge(tweets, on='tweeter_username',how='left')

    affiliation_retweets = affiliation_tweets.groupby(['username','affiliation']).count().\
        bias.reset_index(drop=False).\
        pivot(index='username',columns='affiliation', values='bias').reset_index(drop=False)
    
    for c in ['Liberal','Moderate','Conservative']:
        if c not in affiliation_retweets.columns:
            affiliation_retweets[c] = 0.0
    affiliation_retweets.fillna(0.0,inplace=True)
    affiliation_retweets['trend'] = affiliation_retweets.apply(lambda row: mean3d(row['Conservative'], row['Moderate'], row['Liberal']), axis=1)
    jumped = affiliation_tweets[(affiliation_tweets.username.isin(affiliation_retweets.username))&(affiliation_tweets.tweet_id.isin(main_tweets.tweet_id))]

    jumped = jumped.merge(affiliation_retweets,on='username', suffixes=['','_username'])
    return jumped


def gen_web_graph(jumped2_grp, theme):
    sns.reset_orig()

    if jumped2_grp.shape[0]==0:
        print(f"{theme} data is empty")
        return

    categories = ['Neutral', 'Negative', 'Positive']
    categories_shares = [c+'_username' for c in categories]
    location = 'ON, QC, BC'
    label_map = {'Conservative':'Conservative', 'Moderate':'Moderate', 'Liberal':'Liberal'}
    cats_map = {'Neutral':'neutral', 'Negative':'neg', 'Positive':'pos'}
    colors = {'Liberal':'r', 'Conservative':'b', 'Moderate':'g'}
    titles = ['All',f"\"{theme}\" Sharer Nodes\nIntra-group", 'Inter-group']

    theta = radar_factory(3, frame='polygon')

    data = jumped2_grp.copy()
    for jump_col in ['jumped_0', 'jumped_username', 'jumped_mix']:
        plt.close()
        fig, ax = plt.subplots(1,3,figsize=(9, 5),subplot_kw=dict(projection='radar'))
        ax = ax.flatten()
        for i,tmp in enumerate([data.copy(), data[data[jump_col]==0].copy(), data[data[jump_col]!=0].copy()]):
            key = 'tweet_id'
            mean_sent = tmp.groupby(['bias']).mean().join(tmp.bias.value_counts().rename('tN')).sort_values(by='tN')
            mean_sent = mean_sent.join(tmp.groupby(['bias'])[categories].std(), lsuffix='', rsuffix='_std')
            # mean_sent = tmp.drop_duplicates([key]).groupby(['bias']).mean().join(tmp.drop_duplicates([key]).bias.value_counts().rename('N')).sort_values(by='N')
            # mean_sent = mean_sent.join(tmp.drop_duplicates([key]).groupby(['bias'])[categories].std(), lsuffix='', rsuffix='_std')

            looped_cats_std = [c for c in mean_sent.columns if '_std' in c]

            looped_cats = categories[:3]# + categories[0:1]
            looped_cats_std = looped_cats_std[:3]# + looped_cats_std[0:1]
            N = len(looped_cats)
            theta = radar_factory(N, frame='polygon')
            for aff in  label_map.keys():
                try:
                    d = mean_sent.loc[aff]
                except KeyError:
                    d = pd.Series([0,0,0,0,0,0,0], index=['N']+looped_cats+['Neutral_std','Negative_std', 'Positive_std'])

                color = colors[aff]
                val = d[looped_cats].rename(cats_map)
                y_err = d[looped_cats_std]
                ax[i].plot(theta, val, color=color, label=f"{label_map[aff]} (N={int(d.N)})")

                ax[i].fill(theta,  val.values + y_err.values, facecolor=color, alpha=0.25)
                ax[i].fill(theta,  val.values - y_err.values, facecolor="white", alpha=1)

                ax[i].set_varlabels(val.index)#looped_cats)
                title_n = ''#' '.join([f"{title_map[k]}={int(v)}" for k,v in n.items()])
                ax[i].set_title(titles[i] + "\n"+title_n)
                # ax[i].legend()
                ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), title="       (N=shared tweets)")#, ncol=3, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(f'../data/2023/triag/{theme}_{jump_col}_2.png', bbox_inches="tight")

def draw_trends(theme, jumped2_grp):
    sns.reset_orig()
    tmp1 = jumped2_grp.drop_duplicates(['tweeter_username','tweet_id','bias','text'])
    tmp1 = tmp1.rename(columns={'bias':'Political Leaning'})[['Political Leaning','Negative', 'Neutral', 'Positive']]
    tmp = tmp1.groupby('Political Leaning').mean()
    std = tmp1.groupby('Political Leaning').std()

    tmp = pd.DataFrame([tmp.loc['Conservative'], tmp.loc['Moderate'], tmp.loc['Liberal']])
    std = pd.DataFrame([std.loc['Conservative'], std.loc['Moderate'], std.loc['Liberal']])
    colormap = {'Neutral':'g', 'Negative':'r', 'Positive':'b'}
    plt.close()
    tmp[['Negative', 'Neutral', 'Positive']].plot(kind='bar', yerr=std, capsize=4, color=colormap, rot=0, title=f"Tweet Sentiment by Political Leaning of Author ({theme})", figsize=(7, 4))
    plt.grid('on', which='minor', axis='y' )
    plt.grid('off', which='major', axis='y' )
    plt.xlabel('Author Political Leaning')
    plt.ylabel('Mean Sentiment')
    plt.tight_layout()
    plt.savefig(f'../data/2023/trends/{theme}.png', bbox_inches="tight")

def tri_cmap():
    def inter_from_256(x):
        return np.interp(x=x,xp=[0,255],fp=[0,1])
    cdict = {
        'blue': (
                (0.0, inter_from_256(144), inter_from_256(144)),
                (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
                (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
                (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
                (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
                (1.0, inter_from_256(69), inter_from_256(69))
                ),
        'green': (
                (0.0, inter_from_256(57), inter_from_256(57)),
                # (1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
                (1 / 5 * 3, inter_from_256(241), inter_from_256(241)),
                (1 / 5 * 3, inter_from_256(219), inter_from_256(256)),
                # (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
                (1.0, inter_from_256(23), inter_from_256(23))
                ),
        'red':(
            (0.0,inter_from_256(64),inter_from_256(64)),
            (1/5*0,inter_from_256(112),inter_from_256(112)),
            # (1/5*2,inter_from_256(230),inter_from_256(230)),
            # (1/5*3,inter_from_256(253),inter_from_256(253)),
            (1/5*5,inter_from_256(244),inter_from_256(244)),
            (1.0,inter_from_256(169),inter_from_256(169))
            ),
    }
    return matplotlib.colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)

def tweet_graphs(theme, jumped2_grp):
    sns.reset_orig()
    tmp = jumped2_grp.copy()
    plt.close()
    fig,ax = plt.subplots()
    sns.heatmap(tmp[['bias', 'trend', 'trend_username', 'jumped', 'jumped_0']].corr(), annot=True)
    # sns.heatmap(jumped2_grp[['sentiment', 'Neutral', 'Positive', 'Negative', 'trend', 'trend_username', 'Conservative_username','Liberal_username','Moderate_username']].corr(), annot=True)
    # sns.heatmap(jumped2_grp[['Neutral', 'Positive', 'Negative', 'trend','jumped', 'Conservative','Liberal','Moderate', 'trend_username', 'jumped_username', 'Conservative_username','Liberal_username','Moderate_username']].corr(), annot=True)
    # sns.heatmap(jumped2_grp[['Neutral', 'Negative', 'trend','jumped', 'Conservative','Liberal','Moderate', 'trend_username', 'jumped_username', 'Conservative_username','Liberal_username','Moderate_username']].corr(), annot=True)
    fig.tight_layout()
    plt.savefig(f'../data/2023/trends/corr_{theme}.png', bbox_inches="tight")
    # plt.show()

    plt.close()
    sns.reset_orig()
    fig,ax = plt.subplots(1, 3, figsize=(10,3), constrained_layout=True)
    # bins = list([i/20 for i in range(21)])

    jumped2_grp[['trend', 'trend_username', 'sentiment']]\
        .rename(columns={'sentiment':'Shared Tweet Sentiment', 'trend':"Author Political Leaning)", 'trend_username':'Sharer Political Leaning'})\
        .hist(ax=ax)
    fig.suptitle(f"Shared Tweets Distributions ({theme})")
    # ax[0].set_ylim(top=7500)
    # ax[1].set_ylim(top=7500)
    # ax[2].set_ylim(top=7500)
    ax[0].set_ylabel('Shares (N)')
    ax[0].set_xticks([0,.5,1])
    ax[0].set_xticklabels(['Conservative', 'Moderate\nx=(bias(n)', 'Liberal'])
    ax[1].set_xticks([0,.5,1])
    ax[1].set_xticklabels(['Conservative', 'Moderate\n(trend(n))', 'Liberal'])
    ax[2].set_xticks([0,.5,1])
    ax[2].set_xticklabels(['Negative','Neutral\nx=sent(e)','Positive'])
    fig.tight_layout()
    plt.savefig(f'../data/2023/trends/shared_distributions_{theme}.png', bbox_inches="tight")


    sns.reset_orig()
    plt.close()
    col0 = 'Negative'
    col1 = 'Positive'
    col2 = 'Neutral'
    col3 = 'sentiment'
    col4 = 'trend'
    x0,x1,y = jumped2_grp[[col0,col1, col2]].values.T
    x0,y = jumped2_grp[[col4, col3]].values.T
    fig,ax = plt.subplots()
    plt.scatter(x0, y, c='r', label=col4)
    z0 = np.polyfit(x0, y, 1)
    p0 = np.poly1d(z0)
    plt.plot(x0, p0(x0), c='r')
    plt.xlabel(col4)
    plt.ylabel(col3)


    plt.legend()
    fig.tight_layout()
    plt.savefig(f'../data/2023/trends/trend_corr_{theme}.png', bbox_inches="tight")
    plt.show()

    def rand_jitter(arr):
        stdev = .05 * (max(arr) - min(arr))
        stdev = 0.01 if stdev==0 else stdev
        return arr + np.random.randn(len(arr)) * stdev


    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')


    col0 = 'sentiment'
    col1 = 'trend'

    plt.close()
    fig,ax = plt.subplots(figsize=(5,4))
    plt.title(f"Political Leanings vs \nSentiment  of Shared Tweets\n({theme})")
    sns.violinplot(ax=ax, data=jumped2_grp, x=col1, y=col0,  inner=None, linewidth=0, saturation=0.8)
    sns.boxplot(data=jumped2_grp, x=col1, y=col0, saturation=0.5, width=0.1,
                palette='rocket', boxprops={'zorder': 2}, ax=ax)

    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['Negative (0.0)','Neutral (0.5)', 'Positive (1.0)'])
    ax.set_xticklabels(['Conservative\nAuthors', 'Moderate\nAuthors', 'Liberal\nAuthors'])
    plt.ylabel('Sentiment of shared Tweets (trend(n))')
    plt.xlabel('Political Leaning of shared tweets (sent(e))')
    plt.grid('on', which='minor', axis='y' )
    plt.grid('on', which='major', axis='y' )
    fig.tight_layout()
    plt.savefig(f'../data/2023/trends/trend_x_sentiment_outliers_{theme}.png', bbox_inches="tight")




def main():

    tmp_jumped = []
    for location in LOCATIONS:
        if TOPIC == 'COVID':
            tmp = get_bias(location)
        elif TOPIC == 'Healthcare':
            tmp = get_bias2(location)
        tmp_jumped.append(tmp)
    jumped = pd.concat(tmp_jumped)

    sentiments1 = []
    for file in glob.glob(DATA_PATH_IN):
        print(file)
        tmp = pd.read_csv(file).drop(columns=['Unnamed: 0'], errors='ignore')
        tmp[['sentiment_1', 'sentiment_2','sentiment_3']] = tmp[['sentiment_1', 'sentiment_2','sentiment_3']].apply(lambda row: pd.Series([s.title() for s in row.values]))
        tmp['file'] = os.path.basename(file)
        sentiments1.append(tmp)
    sentiments1 = pd.concat(sentiments1).reset_index(drop=True).drop_duplicates()
    sentiments1 = sentiments1.rename(columns={'left':'Liberal', 'right':'Conservative', 'mid':'Moderate'})

    profiles = []
    for file in glob.glob('../data/output_*_authors_db.csv'):
        tmp = pd.read_csv(file).drop(columns=['Unnamed: 0'], errors='ignore')
        profiles.append(tmp)
    profiles = pd.concat(profiles).reset_index(drop=True).drop_duplicates()
    profiles['username'] = profiles['username'].apply(lambda name: name.lower())

    if TOPIC == 'COVID':
        sentiments2 = []
        for file in glob.glob('./sentiment/retweets*.csv'):
            tmp = pd.read_csv(file).drop(columns=['Unnamed: 0'], errors='ignore')
            tmp[['sentiment_1', 'sentiment_2','sentiment_3']] = tmp[['sentiment_1', 'sentiment_2','sentiment_3']].apply(lambda row: pd.Series([s.title() for s in row.values]))
            sentiments2.append(tmp)
        sentiments2 = pd.concat(sentiments2).reset_index(drop=True).drop_duplicates()
        sentiments2 = sentiments2.merge(profiles, on=['author_id'], how='left')
        sentiments2['tweeter_username'] = sentiments2['username']
        sentiments2 = sentiments2.merge(sentiments1[['tweeter_username', 'bias', 'Liberal', 'Moderate', 'Conservative']].drop_duplicates(), on=['tweeter_username'], how='left')
        sentiments = pd.concat([sentiments1, sentiments2])
        col_subset = ['author_id','tweeter_username', 'username', 'bias', 'Liberal', 'Moderate', 'Conservative']
        sentiments.dropna(subset='bias').tweet_id.value_counts()
        uniq_indx = (sentiments.sort_values(by=['tweet_id','tweeter_username'], na_position='last').dropna(subset=col_subset)
                    .drop_duplicates(subset='tweet_id', keep='first')).index

        # save unique records
        sentiments = sentiments.loc[uniq_indx]
    elif TOPIC == 'Healthcare':
        sentiments = sentiments1.copy()
        sentiments.rename(columns={'id':'tweet_id'}, inplace=True)


    sentiments = sentiments.merge(jumped[['tweet_id','tweeter_username']].drop_duplicates(), on='tweet_id', how='inner')
    jumped = jumped.merge(sentiments[['tweet_id']].drop_duplicates(), on='tweet_id', how='inner')
    

    sentiments = sentiments.drop_duplicates(['author_id', 'tweet_id'], keep='first')
    sentiments.pivot(index='tweet_id',columns=['sentiment_1', 'sentiment_2','sentiment_3'], values=['score_1','score_2','score_3'])
    for col in ['Neutral', 'Positive', 'Negative']:
        sentiments[col] = sentiments.apply(lambda row: row[f"score_{row[['sentiment_1','sentiment_2','sentiment_3']].tolist().index(col)+1}"], axis=1)
    sentiments = sentiments.drop(columns=['sentiment_1', 'sentiment_2','sentiment_3','score_1','score_2','score_3'])
    sentiments['sentiment'] = sentiments.apply(lambda row: mean3d(row['Negative'], row['Neutral'], row['Positive']), axis=1)
    contents = sentiments[['tweet_id']+['text','Neutral', 'Positive', 'Negative', 'sentiment']].copy()


    jumped = jumped.merge(contents[['tweet_id']+['sentiment', 'Neutral', 'Positive', 'Negative']], on='tweet_id',how='left')
    jumped.groupby(['username']).mean().join(jumped.username.value_counts().rename('N')).sort_values(by='N')
    jumped.drop_duplicates(['tweet_id']).groupby(['bias']).mean().join(jumped.drop_duplicates(['tweet_id']).bias.value_counts().rename('N')).sort_values(by='N')

    

    jumped_no_sent = jumped[pd.isnull(jumped.Neutral)].drop_duplicates(['tweet_id'])
    jumped2 =  jumped.dropna(subset=['sentiment','Neutral', 'Positive', 'Negative']).copy()

    jumped2 = jumped2.merge(contents[['tweet_id','text']], on='tweet_id',how='left')

    # find jumpers (inter-group vs intra-group)
    jumped2['jumped'] = 0
    jumped2['jumped_username'] = 0
    jumped2['jumped_mix'] = 0

    jumped2.loc[jumped2[(jumped2['trend']>=0.5)&(jumped2['Negative']>jumped2['Neutral'])].index, 'jumped'] = -1
    jumped2.loc[jumped2[(jumped2['trend']<0.5)&(jumped2['Negative']<jumped2['Neutral'])].index, 'jumped'] = 1
    jumped2.loc[jumped2[(jumped2['trend_username']>=1/2) & ((jumped2['Negative']>jumped2['Neutral'])&(jumped2['Negative']>jumped2['Positive']))].index, 'jumped_username'] = -1
    jumped2.loc[jumped2[(jumped2['trend_username']<1/2)  & ((jumped2['Negative']<jumped2['Neutral'])&(jumped2['Negative']<jumped2['Positive']))].index, 'jumped_username'] = 1
    jumped2.loc[jumped2[(jumped2['trend']>=1/2)&(jumped2['sentiment']<1/3)].index, 'jumped_mix'] = -1
    jumped2.loc[jumped2[(jumped2['trend']<1/2)&(jumped2['sentiment']>=1/3)].index, 'jumped_mix'] = 1

    jumped2['jumped_mix_0'] = jumped2.apply(lambda row: int(row['jumped_mix']!=0), axis=1)
    jumped2['jumped_0'] = jumped2.apply(lambda row: int(row['jumped']!=0), axis=1)
    jumped2['jumped_username_0'] = jumped2.apply(lambda row: int(row['jumped_username']!=0), axis=1)

    jumped2[(jumped2.jumped==0)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]
    jumped2[(jumped2.jumped==1)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]
    jumped2[(jumped2.jumped==-1)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]

    jumped2[(jumped2.jumped_username==0)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]
    jumped2[(jumped2.jumped_username==1)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]
    jumped2[(jumped2.jumped_username==-1)&(~pd.isnull(jumped2.text))][['username','tweet_id','Liberal','Moderate','Conservative','trend','Neutral','Positive','Negative','text','jumped']]


    if GEN_GRAPHS:

        if TOPIC == 'COVID':
            jumped2_grp = jumped2.copy()
            theme = 'COVID19'
            draw_trends(jumped2_grp=jumped2_grp, theme = theme)
            tweet_graphs(jumped2_grp=jumped2_grp, theme = theme)
            gen_web_graph(jumped2_grp=jumped2_grp, theme = theme)

        elif TOPIC == 'Healtcare':
            jumped2 = jumped2.merge(pd.read_csv(DATA_PATH_IN).rename(columns={'id':'tweet_id'})[['tweet_id','themes', 'theme']], on='tweet_id', how='inner')
            jumped2['themes2'] = jumped2.themes.apply(lambda themes: eval(themes) if themes==themes else [])
            themes_list = set(flatten([[t for t in tt] for tt in jumped2.themes2]))
            # for theme,jumped2_grp in jumped2.groupby('theme'):
            theme_ids = {}
            for theme in tqdm.tqdm(themes_list, total=len(themes_list), desc='Assign theme 1'):
                theme_ids[theme] = [ij for ij,j in jumped2.iterrows() if theme in j.themes2]

            ################
            # apply category renaming/regrouping
            theme_rename = {
                'Ontario Government and Community Services'     : 'Politics (Local and Federal)',
                'Federal Government and Metro Vancouver'        : 'Politics (Local and Federal)',
                "Doug Ford's Local Party Updates"               : "Misc Topics",
                "Indigenous Women's Community Stories and Issues"   : "Minority Issues",
                "Toronto Community School Student Life"         : 'Politics (Local and Federal)',
                "Public Health Care in Canada"                  : "Public Health and Politics",
                "Healthcare Services and Crisis Management"     : "Public Health and Politics",
                "Canadian Covid History and Risks"              : "Public Health and Politics",
                "Canadian National Vaccine Plan"                : "Public Health and Politics",
            }
            jumped2['themes3'] = jumped2.themes.apply(lambda themes: [] if themes==themes else [])
            for idx,row in jumped2.iterrows():
                for k,v in theme_rename.items():
                    if k in row['themes2']:
                        jumped2.at[idx, 'themes3'].append(v)

            theme_ids3 = {}
            for theme in tqdm.tqdm(theme_rename.values(), total=len(theme_rename.keys()), desc='Assign Theme 3'):
                theme_ids3[theme] = [ij for ij,j in jumped2.iterrows() if theme in j.themes3]

            themes_idx = set(flatten([ij for ij,j in jumped2.iterrows() if len(j.themes3)>0]))
            no_themes_idx = set(flatten([ij for ij,j in jumped2.iterrows() if len(j.themes3)==0]))

            for theme, idxs in theme_ids3.items():
                jumped2_grp = jumped2.loc[idxs]
                print(theme, jumped2_grp.shape[0])


            # generate graphs
            for theme, idxs in theme_ids3.items():
                try:
                    jumped2_grp = jumped2.loc[idxs]

                    print(theme, jumped2_grp.shape[0])
                    if jumped2_grp.shape[0]<100:
                        continue
                    gen_web_graph(jumped2_grp=jumped2_grp, theme = theme)
                    draw_trends(jumped2_grp=jumped2_grp, theme = theme)
                    tweet_graphs(jumped2_grp=jumped2_grp, theme = theme)
                except Exception as e:
                    print(e)
            jumped2_grp = jumped2.copy()
            theme = 'Healthcare'
            gen_web_graph(jumped2_grp=jumped2_grp, theme = theme)
            draw_trends(jumped2_grp=jumped2_grp, theme = theme)
            tweet_graphs(jumped2_grp=jumped2_grp, theme = theme)

            jumped2_grp = jumped2.loc[no_themes_idx].copy()
            theme = 'Other Topics'
            gen_web_graph(jumped2_grp=jumped2_grp, theme = theme)
            draw_trends(jumped2_grp=jumped2_grp, theme = theme)
            tweet_graphs(jumped2_grp=jumped2_grp, theme = theme)





# # plt.show()

# jumped2[jumped2.bias!=jumped2.affiliation]
# jumped2[['bias', 'affiliation']]

# # corr = jumped2[['trend']+categories].corr()
# tmp = jumped2.drop_duplicates(['tweeter_username','tweet_id','bias','text'])[['trend', 'Liberal','Moderate', 'Conservative', 'Neutral', 'Positive', 'Negative']].copy()
# tmp = tmp[tmp.jumped==1]
# corr = tmp.corr()
# plt.close()
# sns.heatmap(corr, annot=True,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns)
# plt.title(f"Correlation between tweet sentiments")
# plt.show()



def get_tweets(username):
    username = 'sunlorrie'
    import requests

    url = "https://twttrapi.p.rapidapi.com/user-tweets"

    querystring = {"username":username}
    querystring = {"username":"elonmusk"}
    headers = {
        "X-RapidAPI-Key": "6baea2507emshcff10586cb3d908p17dbe7jsn3daf5710e7a2",
        "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.json())



def get_tweets(username):
    username = 'sunlorrie'
    import stweet as st    
    def try_user_scrap(username):
        user_task = st.GetUsersTask([username])
        output_json = st.JsonLineFileRawOutput('output_raw_user.jl')
        output_print = st.PrintRawOutput()
        res = st.GetUsersRunner(get_user_task=user_task, raw_data_outputs=[output_print, output_json]).run()

    def try_tweet_by_id_scrap(tweet_id):
        tweet_id = 1634911959068835840
        id_task = st.TweetsByIdTask(str(tweet_id))
        output_json = st.JsonLineFileRawOutput('output_raw_id.jl')
        output_print = st.PrintRawOutput()
        res = st.TweetsByIdRunner(tweets_by_id_task=id_task, raw_data_outputs=[output_print, output_json]).run()
