import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np
import matplotlib
from scipy import stats
import csv
import pickle
import math


################## prepare data

"""
retweets = {}
tweets = {}
first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT
no_of_weeks = 52*8
no_of_bins = 52*8
daterange = pd.date_range('2012-01-01', periods=52*8, freq='W')
ts = pd.Series(range(len(daterange)), index=daterange)

df = pd.read_pickle('./Gert_large/sample3/dataframe_all_sample3_05-2019_anonymized')
print(df.keys())

for index, row in df.iterrows():
	#if index > 100:
	#	break
	#print(row['time'])
	df1 = pd.DataFrame(row['time'], columns=['date'])
	df1.index = pd.to_datetime(df1['date'])
	df1['n'] = 1
	#if sum(df1['n']) < 1:
	#	continue
	df_weekly = df1.n.resample('W').sum()
	user = [0 for i in range(no_of_bins)]
	for index1, row1 in df_weekly.iteritems():
		index_no = ts[index1]
		user[index_no] = row1
	#plt.plot(user)
	#plt.show()
	if row['rt_id_anonymous'] != "nope":
		retweets[row['id_anonymous']] = user
	else:
		tweets[row['id_anonymous']] = user

f = open('weekly_tweets_sample3.pckl', 'wb')
pickle.dump(tweets, f)
f.close()
f = open('weekly_retweets_sample3.pckl', 'wb')
pickle.dump(retweets, f)
f.close()

"""
######################## plotting

f = open('./Gert_prepared/data/sample3/weekly_tweets_sample3.pckl', 'rb')
tweets = pickle.load(f)
f = open('./Gert_prepared/data/sample3/weekly_retweets_sample3.pckl', 'rb')
retweets = pickle.load(f)

colors = matplotlib.cm.viridis(np.linspace(0, 1, 8))
no_of_weeks = 52
####cluster
count = 0
for clusteri in [4,0,7,3,1,6,5,2]:
#for clusteri in range(8):
	ids = list(pd.read_pickle('./Gert_large/sample3/cluster_all_data_{0}.pkl'.format(clusteri))['id_anonymous'])
	mean_traj = [[] for i in range(no_of_weeks)]
	print(len(ids))
	for user_id in ids:
		try:
			user = retweets[user_id]
			trimmed = np.trim_zeros(user)
			#trimmed = [el/max(trimmed) for el in trimmed]
			if len(trimmed) > no_of_weeks:
				trimmed = trimmed[:no_of_weeks]
			elif len(trimmed) <= no_of_weeks:
				continue
			#trimmed = np.pad(trimmed, (0,no_of_weeks-len(trimmed)), 'constant')
			for time in range(no_of_weeks):
				mean_traj[time].append(trimmed[time])
		except KeyError:
			continue

	color = colors[count]
	count += 1
	mean_vals = np.array([np.mean(x) for x in mean_traj])
	err_vals = np.array([stats.sem(x) for x in mean_traj])
	plt.plot([i for i in range(len(mean_vals))], mean_vals, label='cohort '+str(clusteri+1), color=color)
	plt.fill_between([i for i in range(len(mean_vals))], mean_vals-err_vals, mean_vals+err_vals, alpha=0.4, color=color)
	#plt.plot([np.mean(el) for el in mean_traj], color='black')
plt.legend()
plt.xlabel("week")
plt.ylabel("av. tweets/week")
plt.show()
