import pandas as pd
import numpy as np
import pickle
import time
import datetime
import itertools
from sklearn.cluster import KMeans


first_post = 1325376000 # unix time stamp 1.1.2012
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT
no_of_bins = 2709 #total days from 1.1.12 to 1.6.2019
clusters = 8

bins = np.linspace(first_post, last_post,no_of_bins) #each day one bin

df = pd.read_pickle('data/dataframe_all_sample2_04-2019_anonymized') #load df

#delete all columns not used for setting up clusters
del df['rt_id_anonymous']
del df['hashtags']
del df['url']


df = df.groupby('id_anonymous').agg({'time':sum}).reset_index() #aggregate posts


user_ids = df['id_anonymous'].values[:200000] #get all ids of actively tweeting users (all below are passive retweets)


ratio_active_days = []
for user in range(200000):

	posttimes = sorted(list([time.mktime(datetime.datetime.strptime(i, "%a %b %d %H:%M:%S +0000 %Y").timetuple()) for i in df['time'][user]])) # sort all posttimes
	hist = np.histogram(posttimes,bins=bins) #bin posttimes
	k = (hist[0]!=0).argmax(axis=0)
	act_arr = hist[0][k:]
	nonzero_elements = len(act_arr[act_arr!=0]) 
	active_days = nonzero_elements/float(len(act_arr)) #compute ratio of active days
	ratio_active_days.append(active_days)


	
kmeans = KMeans(n_clusters = clusters) 
kmeans.fit(ratio_active_days.reshape(-1,1)) #kmeans clustering of all users



for i in range(clusters):
	#set up df for each kmeans (user_type) cluster
	indices = np.where(kmeans.labels_ == i)
	df_cluster1 = pd.DataFrame(data={'id_anonymous': range(len(user_ids[indices[0]])), 'time': range(len(user_ids[indices[0]]))})
	df_cluster1['time'] = df['time'].astype('object')
	
	cnt=0
	for user_id in user_ids[indices[0]]:
		#fill in ids and post times of users to df
		df_cluster1.at[cnt, 'time'] = df[df['id_anonymous'] == user_id]['time'].tolist()
		df_cluster1.at[cnt, 'id_anonymous'] = user_id
		cnt+=1

	df_cluster1.to_pickle("data/cluster_activity_all_data_%s.pkl"%(i))

