import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np
import matplotlib
from scipy import stats
import networkx as nx
import csv
from scipy.stats.stats import pearsonr 
from scipy.stats.stats import spearmanr 
import pickle
import math

colors = matplotlib.cm.inferno(np.linspace(0, 1, 7))

active_retweets = {}
passive_retweets = {}
tweets = {}
first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT
no_of_weeks = 52*8
no_of_bins = 52*8
daterange = pd.date_range('2012-01-01', periods=52*8, freq='W')
ts = pd.Series(range(len(daterange)), index=daterange)


def jaccard2(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    num_intersection = len(set1 & set2)
    return num_intersection / float(len(set1) + len(set2) - num_intersection)

f = open('./weekly_active_hashtags.pckl', 'rb')
tweets = pickle.load(f)

####cluster
count = 0
for clusteri in range(1,7):
	distinct_hashtags = []
	ids = list(pd.read_pickle('./cohort_{0}_all_data.pkl'.format(clusteri))['id_anonymous'])
	#ids = list(pd.read_pickle('./Gert_large/sample2/cluster_all_data_{0}.pkl'.format(clusteri))['id_anonymous'])
	corr_vals = [[] for h in range(26)]
	for user_id in ids:
		try:
			if len(tweets[user_id]) > 0: #and sum(retweets[user_id]) > 0:
				
				tweets_m = list(tweets[user_id])
				tweets_1 = []
				for item in tweets_m:
					if item == 0:
						tweets_1.append(0)
					elif len(item) > 0:
						tweets_1.append(item)
					else:
						tweets_1.append(0)

				tweets_1 = tweets_1[(clusteri)*52:(clusteri+1)*52]
				distinct_hashtags.append(len([y for x in tweets_1 if x!= 0 for y in x]))
				
				match_list = []
				for lag in range(1, 27):
					a = [0 for j in range(26)] + tweets_1 + [0 for j in range(26)]
					b = [0 for j in range(26+lag)] + tweets_1 + [0 for j in range(26-lag)]
					matches = []
					for i, j in zip(a, b):
						if i !=0 and j != 0:
							matches.append(jaccard2(i, j))
							
					if sum(matches) > 0:

						match_list.append(np.mean(matches))
					else:
						match_list.append(np.nan)
					
				for i, val in enumerate(match_list):
					corr_vals[i].append(val)

		except KeyError:
			continue
	print(np.mean(distinct_hashtags))
	color = colors[count]
	mean_vals = np.array([np.nanmean(x) for x in corr_vals])
	print(mean_vals)
	
	err_vals = np.array([stats.sem(x, nan_policy='omit') for x in corr_vals])
	plt.plot([i for i in range(1, 27)], mean_vals, label="cohort "+str(2012+clusteri), color=color)
	plt.fill_between([i for i in range(1, 27)], mean_vals-err_vals, mean_vals+err_vals, alpha=0.4, color=color)
	plt.xlabel("$time lag [weeks]$")
	plt.ylabel("$av. cross-correlation$")
	plt.tight_layout()
	count += 1
plt.legend()
plt.show()
