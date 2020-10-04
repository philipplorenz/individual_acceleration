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

colors = matplotlib.cm.viridis(np.linspace(0, 1, 8))

active_retweets = {}
passive_retweets = {}
tweets = {}
first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT
no_of_weeks = 52*8
no_of_bins = 52*8
daterange = pd.date_range('2012-01-01', periods=52*8, freq='W')
ts = pd.Series(range(len(daterange)), index=daterange)

"""
#df = pd.read_pickle('./dataframe_passive_retweet_network_all_05-2019_anonymized')
df = pd.read_pickle('./dataframe_all_05-2019_anonymized')
#df = df['time'].apply(lambda x: x.split(','))
print(df)
print(df.keys())

for index, row in df.iterrows():
	#if index > 1000:
	#	break
	#print(row['time'])
	df1 = pd.DataFrame(row['time'], columns=['date'])
	df1.index = pd.to_datetime(df1['date'])
	df1['n'] = 1
	if sum(df1['n']) < 1:
		continue
	df_weekly = df1.n.resample('W').sum()
	#print(df_weekly)
	user = [0 for i in range(no_of_bins)]
	for index1, row1 in df_weekly.iteritems():
		index_no = ts[index1]
		user[index_no] = row1
	#plt.plot(user)
	#plt.show()
	if row['rt_id_anonymous'] != "nope":
		active_retweets[row['id_anonymous']] = user
	#else:
	#	tweets[row['id_anonymous']] = user

f = open('weekly_active_retweets.pckl', 'wb')
pickle.dump(active_retweets, f)
f.close()


df = pd.read_pickle('./Gert_large/sample2/dataframe_passive_retweets_sample2_04-2019_anonymized')
#df = pd.read_pickle('./dataframe_all_05-2019_anonymized')
#df = df['time'].apply(lambda x: x.split(','))
print(df)
#print(df.keys())

#df1 = df

edge_list1 = df.values

for row in edge_list1:
	#print(row)
	if len(row[3]) < 2:
		continue
	#if index > 1000:
	#	break
	#print(row['time'])
	df1 = pd.DataFrame(row[3], columns=['time'])
	df1.index = pd.to_datetime(df1['time'])
	df1['n'] = 1
	if sum(df1['n']) < 1:
		continue
	df_weekly = df1.n.resample('W').sum()
	#print(df_weekly)
	user = [0 for i in range(no_of_bins)]
	for index1, row1 in df_weekly.iteritems():
		index_no = ts[index1]
		user[index_no] = row1
	#plt.plot(user)
	#plt.show()
	passive_retweets[row[0]] = user

f = open('./Gert_prepared/data/sample2/weekly_passive_tweets.pckl', 'wb')
pickle.dump(passive_retweets, f)
f.close()

####cluster
count = 0
for clusteri in [0, 5, 2, 6, 1, 7, 3, 4]:
	ids = list(pd.read_pickle('./cluster_all_data_{0}.pkl'.format(clusteri))['id_anonymous'])
	corr_vals = []
	for user_id in ids:
		try:
			if sum(tweets[user_id]) > 0 and sum(retweets[user_id]) > 0:
				tweet_mean = np.mean([elem for elem in tweets[user_id] if elem > 0])
				tweets_m = [el - tweet_mean if el > 0 else el for el in tweets[user_id]]
				retweet_mean = np.mean([elem for elem in retweets[user_id] if elem > 0])
				retweets_m = [el - retweet_mean if el > 0 else el for el in retweets[user_id]]
				#plt.scatter(tweets[user_id], retweets[user_id])
				if not math.isnan(pearsonr(tweets_m, retweets_m)[0]): 
					corr_vals.append(pearsonr(tweets_m, retweets_m))
				#tweets[user_id].corr(retweets[user_id].shift(0))
		except KeyError:
			continue

	value_list = [el[0] for el in corr_vals]
	print(np.mean(value_list))
	in_hist = np.histogram(value_list, 30)
	y = in_hist[0]/float(len(value_list))
	err = np.sqrt(y)
	err[err >= y] = y[err >= y] - 1e-2
	#plt.figure(figsize=(6,4))
	plt.errorbar(in_hist[1][:-1], y, fmt="o", color = colors[count], label=clusteri)
	plt.xlabel("$Corr(tweets, retweets$")
	plt.ylabel("$P(Corr)$")
	plt.legend()
	plt.tight_layout()
	count += 1
plt.show()
"""
f = open('./Gert_prepared/data/sample2/weekly_tweets_sample2.pckl', 'rb')
tweets = pickle.load(f)
f = open('./Gert_prepared/data/sample2/weekly_retweets_sample2.pckl', 'rb')
active_retweets = pickle.load(f)
f = open('./Gert_prepared/data/sample2/weekly_passive_tweets.pckl', 'rb')
passive_retweets = pickle.load(f)

retweets = {**active_retweets, **passive_retweets}

####cluster
count = 0
for clusteri in [4,1,6,3,0,5,2,7]:
#for clusteri in range(8):
	ids = list(pd.read_pickle('./Gert_large/sample2/cluster_all_data_{0}.pkl'.format(clusteri))['id_anonymous'])
	corr_vals = [[] for h in range(9)]
	for user_id in ids:
		try:
			if sum(tweets[user_id]) > 0 and sum(retweets[user_id]) > 0:
				#tweets_m = list(tweets[user_id])
				#retweets_m = list(retweets[user_id])
				tweet_mean = np.mean([elem for elem in tweets[user_id] if elem > 0])
				tweets_m = [el - tweet_mean if el > 0 else el for el in tweets[user_id]]
				retweet_mean = np.mean([elem for elem in retweets[user_id] if elem > 0])
				retweets_m = [el - retweet_mean if el > 0 else el for el in retweets[user_id]]
				#plt.scatter(tweets[user_id], retweets[user_id])
				for lag in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
					corr_value = pearsonr([0, 0, 0, 0]+tweets_m+[0, 0, 0, 0], [0 for i in range(4+lag)]+retweets_m+[0 for i in range(4-lag)])
					if not math.isnan(corr_value[0]): 
						corr_vals[lag+4].append(corr_value[0])
					#tweets[user_id].corr(retweets[user_id].shift(0))
		except KeyError:
			continue

	color = colors[count]
	mean_vals = np.array([np.mean(x) for x in corr_vals])
	print(mean_vals[3], mean_vals[5])
	err_vals = np.array([stats.sem(x) for x in corr_vals])
	plt.plot([-4, -3, -2, -1, 0, 1, 2, 3, 4], mean_vals, label='cluster '+str(clusteri+1), color=color)
	plt.fill_between([-4, -3, -2, -1, 0, 1, 2, 3, 4], mean_vals-err_vals, mean_vals+err_vals, alpha=0.4, color=color)
	plt.xlabel("$time lag$")
	plt.ylabel("$av. correlation$")
	plt.legend()
	plt.tight_layout()
	count += 1
plt.show()
