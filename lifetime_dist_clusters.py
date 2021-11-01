import pandas as pd
import numpy as np
import pickle
import time
import datetime
import itertools

clusters = 8



first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT





starting_day_all_clusters = []
users_in_cluster = []

for j in range(clusters):
	cnt = 0
	
	starting_all = []
	df = pd.read_pickle('data/cluster_all_data_%s.pkl'%(j)) #load user type df
	
	for user in range(len(df.index)):
		posttimes = sorted(list([time.mktime(datetime.datetime.strptime(i, "%a %b %d %H:%M:%S +0000 %Y").timetuple()) for i in df['time'].iloc[user][0]])) #sort posttimes of user

		if last_post - posttimes[0] > 31536000:	# only consider users who have tweeted a full year(joined before June 2018)
			starting_day = posttimes[0]
			
			if starting_day > first_post+31536000: # only consider users who have not been active in 2012 (as a proxy that they have not joined before)
				starting_all.append(starting_day) #append the starting date of the filtered users
				cnt+=1
	
	users_in_cluster.append(cnt)
	

	starting_day_all_clusters.append(starting_all)
	starting_all = np.array(starting_all)

with open("starting_day_all_clusters_without_cohort1and8.txt", "wb") as fp:   #Pickling
	pickle.dump(starting_day_all_clusters, fp)

with open("starting_day_all_clusters_without_cohort1and8.txt", "rb") as fp:   # Unpickling
	starting_day_all_clusters = pickle.load(fp)


#plot as you wish
