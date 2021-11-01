import pandas as pd
import numpy as np
import pickle
import time
import datetime

first_post = 1325376000 # unix time stamp 1.1.2012
end_of_2012 =  1356998400 # ...
end_of_2013 =  1388534400
end_of_2014 =  1420070400
end_of_2015 =  1451606400
end_of_2016 =  1483228800
end_of_2017 =  1514764800
end_of_2018 =  1546300800
end_of_may_2019 =  1559347200


df = pd.read_pickle('data/dataframe_all_sample2_04-2019_anonymized') # load DF

del df['rt_id_anonymous'] #delete column which are not used for setting up cohorts
del df['hashtags']
del df['url']

df = df.groupby('id_anonymous').agg({'time':sum}).reset_index() # aggregate timestamps


cohort1 = []
cohort2 = []
cohort3 = []
cohort4 = []		
cohort5 = []
cohort6 = []
cohort7 = []
cohort8 = []



for user in range(200000):
	posttimes = sorted(list([time.mktime(datetime.datetime.strptime(i, "%a %b %d %H:%M:%S +0000 %Y").timetuple()) for i in df['time'].iloc[user]]))
	
	timestamp = posttimes[0] # identify first post
	
	#Bin timestamp to year
	if timestamp > first_post and timestamp < end_of_2012:
		cohort1.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2012 and timestamp < end_of_2013:
		cohort2.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2013 and timestamp < end_of_2014:
		cohort3.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2014 and timestamp < end_of_2015:
		cohort4.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2015 and timestamp < end_of_2016:
		cohort5.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2016 and timestamp < end_of_2017:
		cohort6.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2017 and timestamp < end_of_2018:
		cohort7.append(df['id_anonymous'].iloc[user])
	if timestamp > end_of_2018 and timestamp < end_of_may_2019:
		cohort8.append(df['id_anonymous'].iloc[user])

cohorts=[]
cohorts.append(cohort1)

cohorts.append(cohort2)

cohorts.append(cohort3)

cohorts.append(cohort4)

cohorts.append(cohort5)

cohorts.append(cohort6)

cohorts.append(cohort7)

cohorts.append(cohort8)


for i in range(8):
	#setup cohort DF
	df_cluster1 = pd.DataFrame(data={'id_anonymous': range(len(cohorts[i])), 'time': range(len(cohorts[i]))})
	df_cluster1['time'] = df['time'].astype('object')
	
	
	cnt = 0
	for user_id in cohorts[i]:
		#allocate user to row in column
		df_cluster1.at[cnt, 'time'] = df[df['id_anonymous'] == user_id]['time'].tolist()
		df_cluster1.at[cnt, 'id_anonymous'] = user_id

		cnt+=1


	df_cluster1.to_pickle("data/cohort_%s_all_data.pkl"%i)

