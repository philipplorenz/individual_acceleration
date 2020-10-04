import pandas as pd
import matplotlib.pyplot as plt	
import numpy as np
import matplotlib
from scipy import stats
import networkx as nx
import csv
from jaccard_index.jaccard import jaccard_index
import pickle

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / len(list1)


#df = pd.read_pickle('./Gert_prepared/data/sample3/dataframe_passive_retweets_sample3_05-2019_anonymized')
#print(df)
#print(df.keys())
colors = matplotlib.cm.viridis(np.linspace(0, 1, 8))

overlap_values = []
count_n = 0
for clusti in [4,1,6,3,0,5,2,7]:
#for clusti in range(8):
	ids = list(pd.read_pickle('./Gert_large/sample2/cluster_all_data_{0}.pkl'.format(clusti))['id_anonymous'])
	overlap_values.append([])
	"""
	own_hashtags = {}
	other_hashtags = {}

	count_n = 0

	#df['time'] = df['time'].apply(lambda x: len(x.split(",")))
	for index, row in df.iterrows():
		if len(max(row['hashtags'],key=len)) > 0:
			#print(row['hashtags'])
			if row['id'] in ids:
				if row['rt_id'] != "nope":
					own_hashtags[row['id']] = []
					for tags in row["hashtags"]:
						if len(tags) > 0:
							own_hashtags[row['id']] += tags
				else:
					other_hashtags[row['id']] = []
					for tags in row["hashtags"]:
						if len(tags) > 0:
							other_hashtags[row['id']] += tags
	print(own_hashtags)
	print(other_hashtags)
	
	f = open("own_hashtags_sample3_cluster_{0}.pkl".format(clusti),"wb")
	pickle.dump(own_hashtags,f)
	f.close()
	f = open("other_hashtags_sample3_cluster_{0}.pkl".format(clusti),"wb")
	pickle.dump(other_hashtags,f)
	f.close()
	"""

	with open("./Gert_prepared/data/sample2/own_hashtags_sample2_cluster_{0}.pkl".format(clusti), 'rb') as handle:
    		own_hashtags = pickle.load(handle)	
	with open("./Gert_prepared/data/sample2/other_hashtags_sample2_cluster_{0}.pkl".format(clusti), 'rb') as handle:
    		other_hashtags = pickle.load(handle)	


	for idi in ids:
		try:
			print(own_hashtags[idi], other_hashtags[idi])
			print(jaccard_similarity(own_hashtags[idi], other_hashtags[idi]))
			#if jaccard_similarity(own_hashtags[idi], other_hashtags[idi]) > 0:
			overlap_values[count_n].append(jaccard_similarity(own_hashtags[idi], other_hashtags[idi]))

		except KeyError:
			continue
	
	print(len(overlap_values[count_n]))

	in_hist = np.histogram(overlap_values[count_n], 10)
	print(np.mean(overlap_values[count_n]))
	y = in_hist[0]/float(len(overlap_values[count_n]))
	err = np.sqrt(y)
	err[err >= y] = y[err >= y] - 1e-2

	#plt.figure(figsize=(6,4))
	plt.errorbar(in_hist[1][:-1], y, fmt="o", color = colors[count_n], label="cluster {}".format(clusti))
	plt.xlabel("$hashtag homophily$")
	plt.ylabel("$P$")
	plt.tight_layout()
	count_n += 1

plt.legend()
plt.show()

plt.boxplot(overlap_values)
plt.show()
