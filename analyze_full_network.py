import pandas as pd
from graph_tool.all import *
import matplotlib.pyplot as plt	
import numpy as np
import matplotlib
from scipy import stats
import networkx as nx
import csv


colors = matplotlib.cm.viridis(np.linspace(0, 1, 8))

"""
#for clusteri in range(8):
#	ids = pd.read_pickle('./cluster_97509_{0}_ids.pkl'.format(clusteri))
count_n = 0
edge_list = []
weight_list = []
#G=nx.DiGraph()
G = Graph()
#nx.set_node_attributes(G, 'cluster', 0.)
weight = G.new_edge_property("int")

df = pd.read_pickle('./Gert_large/sample2/dataframe_all_sample2_04-2019_anonymized')
print(df)
print(df.keys())
#df['time'] = df['time'].apply(lambda x: len(x.split(",")))
activity_dict = {}
for index, row in df.iterrows():
	#if edge[0] in ids or edge[1] in ids:
	#print(edge)
	#if str(year) in edge[2]:
	#	#print(edge)
	#	edge_list.append([edge[0], edge[1], edge[2].count(str(year))])
	if row['rt_id_anonymous'] != "nope":
		#count = 0
		#for timestep in row['time']:
		#	if str(year) in timestep:
		#		count += 1
		#if count > 0: 
		#	edge_list.append([row['id_anonymous'], row['rt_id_anonymous'], count])
		edge_list.append([row['id_anonymous'], row['rt_id_anonymous'], len(row['time'])])
	else:
		#count = 0
		#for timestep in row['time']:
		#	if str(year) in timestep:
		#		count += 1
		#if count > 0: 
		#	activity_list.append([row['id_anonymous'], count])
		activity_dict[row['id_anonymous']] = len(row['time'])

df = pd.read_pickle('./Gert_large/sample2/dataframe_passive_retweets_sample2_04-2019_anonymized')
#df['time'] = df['time'].apply(lambda x: len(x.split(",")))
edge_list1 = df.values

for edge in edge_list1:
	#if str(year) in edge[2]:
	#	#print(edge)
	#	edge_list.append([edge[0], edge[1], edge[2].count(str(year))])
	edge_list.append([edge[0], edge[1], len(edge[3])])
		
#df = pd.read_pickle('./dataframe_retweet_network_05-2019_anonymized')
#df['time'] = df['time'].apply(lambda x: len(x.split(",")))
#edge_list2 = df.values
#for edge in edge_list1:
#	if edge[0] in ids or edge[1] in ids:
#		edge_list.append(edge)

#edge_list = np.concatenate((edge_list1, edge_list2))


node_id = G.add_edge_list(edge_list, hashed=True, eprops=[weight])
print(node_id)
G.vertex_properties['node_id'] = node_id

activity = G.new_vertex_property("int")
G.vertex_properties['activity'] = activity
G.edge_properties['weight'] = weight
G.list_properties()

G = extract_largest_component(G, prune=True)
G.save('my_network_raw_sample2.graphml')

#for user in activity_list:
for v in G.vertices():
	try:
		G.vp.activity[v] = activity_dict[G.vp.node_id[v]]
	except KeyError:
		continue

# Same with edge properties `color` and `weight`
cluster = G.new_vertex_property("int")
G.vertex_properties['cluster'] = cluster

for clusteri in range(8):
	ids = list(pd.read_pickle('./Gert_large/sample2/cluster_all_data_{0}.pkl'.format(clusteri))['id_anonymous'])
	print(ids)
	for v in G.vertices():
		if G.vp.node_id[v] in ids:
			G.vp.cluster[v] = clusteri+1

G.save('my_network_sample2.graphml')
"""
G = load_graph("my_network_sample2.graphml")
count_n = 0
G.list_properties()
d = G.degree_property_map("in", weight=G.ep.weight)   # weight is an edge property map

for clusti in [4,1,6,3,0,5,2,7]:
	
	value_list = []
	for v in G.vertices():
		if G.vp.cluster[v] == clusti+1:
			value_list.append(v.out_degree())
			#d[v] /= float(v.out_degree())
	in_hist = np.histogram(value_list, 50)
	print(np.mean(value_list))
	y = in_hist[0]/float(len(value_list))
	err = np.sqrt(y)
	err[err >= y] = y[err >= y] - 1e-2

	#plt.figure(figsize=(6,4))
	plt.errorbar(in_hist[1][:-1], y, fmt="o", color = colors[count_n], label="cluster {0}".format(clusti))
	plt.xlabel("$k_{out}$")
	plt.ylabel("$NP(k_{out})$")
	plt.tight_layout()
	#plt.show()
	count_n += 1
#id_list = {}
#for node in range(G.num_vertices()):
#	id_list[node_id[node]] = node

#for clusteri in range(8):
#	ids = pd.read_pickle('./cluster_97509_{0}_ids.pkl'.format(clusteri))
#	for node in ids:
#		#print(node, node_id[node])
#		try:
#			idx = id_list[node]
#			cluster[idx] = clusteri
#			print(cluster[idx])
#		except KeyError:
#			continue

#pos = sfdp_layout(G)
#graph_draw(G, pos, output_size=(1000, 1000), vertex_color=[1,1,1,0], vertex_size=1, edge_pen_width=1.2,
#   vcmap=matplotlib.cm.gist_heat_r, output="{0}.pdf".format(year))
# Save graph

#pos = sfdp_layout(G)
#graph_draw(G, pos, vertex_color=cluster)
#plt.show()
#G.add_edges_from(edge_list)
#df2 = pd.read_pickle('./dataframe_retweet_network_05-2019_anonymized')
#for index, row in df2.iterrows():
#	#row["time"] = [to_integer(datetime.strptime(el[:10:4:], '%a %b %d %Y')) for el in row["time"].split(",")]
#	#if str(year) in row["time"]:
#	#if row['id2'] in ids:
#	G.add_node(row['id1'])
#	G.add_node(row['id2'])
#	G.add_edge(row['id1'], row['id2'])
#	G[row['id1']][row['id2']]['weight'] = len(row['time'].split(","))
#	#edge_list.append([row['id1'], row['id2']])

#df["time"].to_timestamp()
#with open('retweet_edge_list_{0}.csv'.format(year), "w", newline="") as f:
#        writer = csv.writer(f)
#        writer.writerows(edge_list)
#df.to_csv('retweet_edge_list_{0}.csv'.format(year), sep=',', index=False)
#Gc = max(nx.connected_component_subgraphs(G), key=len)
#Gc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
#largest_cc = max(nx.connected_components(G), key=len)	
#colors = [Gc.node[n]['cluster'] for n in Gc.nodes()]
#nx.write_graphml(Gc, "full.graphml")
#nx.draw(G)
#nx.draw(G, pos=nx.spring_layout(G), node_color=colors) 
plt.legend()
plt.show()
