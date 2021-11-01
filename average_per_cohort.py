import pandas as pd
from graph_tool.all import *
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

retweets = {}
tweets = {}
first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200# unix timestamp 1.6.2019 0:00 GMT
no_of_weeks = 52
no_of_bins = 52*8
daterange = pd.date_range('2012-01-01', periods=52*8, freq='W')
ts = pd.Series(range(len(daterange)), index=daterange)

f = open('daily_sample2_tweets.pckl', 'rb')
tweets = pickle.load(f)
f = open('daily_sample2_retweets.pckl', 'rb')
retweets = pickle.load(f)

no_of_weeks = 52

####cluster
count = 0
average_tweets = []
age = []
#for clusteri in [6,1,3,4,0,5,2,7]:
for clusteri in range(1,7):
	ids = list(pd.read_pickle('./cohort_{0}_all_data.pkl'.format(clusteri))['id_anonymous'])
	mean_traj = [[] for i in range(8*no_of_weeks)]
	average_tweets.append([])
	print(len(ids))
	for user_id in ids:
		try:
			user = tweets[user_id]
			
			trimmed = np.trim_zeros(user, "f")
			print(len(trimmed))
			
			if len(trimmed) <= no_of_weeks:
				continue
			if len(trimmed) > no_of_weeks:
				trimmed = trimmed[:no_of_weeks]
			trimmed = np.array(trimmed[1:])
		
			average_tweets[-1].append(np.mean(trimmed))
			for time in range(len(trimmed)):
				mean_traj[(clusteri)*52+time].append(trimmed[time])
		except KeyError:
			continue

	color = colors[count]
	count += 1

	mean_vals = np.array([np.mean(x) for x in mean_traj])
	#err_vals = np.array([np.std(x) for x in mean_traj])
	plt.plot([i for i in range(len(mean_vals))], mean_vals, label='generation '+str(2012+clusteri), color=color, lw=2, alpha=1.0)
	#plt.fill_between([i for i in range(len(mean_vals))], mean_vals-err_vals, mean_vals+err_vals, alpha=1., color=color)
	err_vals = np.array([stats.sem(x) for x in mean_traj])
	plt.fill_between([i for i in range(len(mean_vals))], mean_vals-err_vals, mean_vals+err_vals, alpha=0.5, color=color)
	#plt.plot([np.mean(el) for el in mean_traj], color='black')
count = 0
average_tweets = []
age = []
for clusteri in range(1,7):
	ids = list(pd.read_pickle('./cohort_{0}_all_data.pkl'.format(clusteri))['id_anonymous'])
	mean_traj = [[] for i in range(8*no_of_weeks)]
	average_tweets.append([])
	print(len(ids))
	for user_id in ids:
		try:
			user = tweets[user_id]
			
			trimmed = np.trim_zeros(user, "f")
			print(len(trimmed))
			
			if len(trimmed) <= no_of_weeks:
				continue
			trimmed = np.array(trimmed[1:])
		
			average_tweets[-1].append(np.mean(trimmed))
			for time in range(len(trimmed)):
				mean_traj[(clusteri)*52+time].append(trimmed[time])
		except KeyError:
			continue

	color = colors[count]
	count += 1
	
	mean_vals = np.array([np.mean(x) for x in mean_traj])
	plt.plot([i for i in range(len(mean_vals))], mean_vals, color=color, lw=2, alpha=0.3)
	err_vals = np.array([stats.sem(x) for x in mean_traj])
	plt.fill_between([i for i in range(len(mean_vals))], mean_vals-err_vals, mean_vals+err_vals, alpha=0.15, color=color)
plt.legend()
plt.xlabel("week")
plt.ylabel("av. tweets/week")
plt.show()
box = plt.boxplot(average_tweets, showfliers=False, notch=True, patch_artist=True)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()
plt.scatter(age, average_tweets, alpha=0.2)
plt.show()
