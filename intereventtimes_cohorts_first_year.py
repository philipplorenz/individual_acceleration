import pandas as pd
import numpy as np
import pickle
import time
import datetime
import itertools

cohorts = 7 #last cohort joined 2019 does not have had the opportunity to tweet a full consecutive year


#analog computation as for the clusters

first_post = 1325372400 # unix timestamp 1.1.2012 0:00 GMT
last_post = 1559347200	# unix timestamp 1.6.2019 0:00 GMT



interposttimes_all_cohorts = []

users_in_cohort =[]

for j in range(cohorts):
	
	interposttimes_all = []
	df = pd.read_pickle('data/cohort_%s_all_data.pkl'%j)
	
	cnt = 0
	for user in range(len(df.index)):
		posttimes = sorted(list([time.mktime(datetime.datetime.strptime(i, "%a %b %d %H:%M:%S +0000 %Y").timetuple()) for i in df['time'][user][0]]))
		if last_post - posttimes[0] > 31536000:
			login_date = posttimes[0]
			posttimes_first_year = [i for i in posttimes if i-login_date < 31536000]
		
		
			posttimes = sorted(list(posttimes_first_year))
			interposttimes = np.diff(np.array(posttimes))
			
	
			
			
			interposttimes_all.append(interposttimes)
			cnt+=1
	
	
		
	interposttimes_all_cohorts.append(interposttimes_all)
	users_in_cohort.append(cnt)
	print j


np.save('interposttimes_al_cohorts',np.array(interposttimes_all_cohorts))
np.save('users_in_cohort',np.array(users_in_cohort))
	
#plot as you wish
