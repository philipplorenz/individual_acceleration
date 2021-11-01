import pandas as pd
import numpy as np
import pickle
import time
import datetime
import itertools
clusters = 8


first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT



interposttimes_all_clusters = []

users_in_cluster =[]

for j in range(clusters):
	
	interposttimes_all = []
	df = pd.read_pickle('data/cluster_all_data_%s.pkl'%(j))
	
	cnt = 0
	for user in range(len(df.index)):
		posttimes = sorted(list([time.mktime(datetime.datetime.strptime(i, "%a %b %d %H:%M:%S +0000 %Y").timetuple()) for i in df['time'].iloc[user][0]])) #sort posttimes
		login_date = posttimes[0]
		if last_post - login_date > 31536000: #only consider users who have posted at least one year (started before June 2018)
			posttimes_first_year = [i for i in posttimes if i-login_date < 31536000] # only consider first active year
				
			posttimes = sorted(list(posttimes_first_year)) #sort posttimes again (should anyway be already sorted)
			interposttimes = np.diff(np.array(posttimes)) #compute interposttimes
			
	
			
			
			interposttimes_all.append(interposttimes)
			cnt+=1
	
	
		
	interposttimes_all_clusters.append(interposttimes_all)
	users_in_cluster.append(cnt)
	


np.save('interposttimes_al_clusters',np.array(interposttimes_all_clusters))
np.save('users_in_clusters',np.array(users_in_cluster))
	
#plot as you wish
