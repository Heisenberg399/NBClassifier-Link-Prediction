#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import os
from tqdm import tqdm 
import collections
import numpy as np
from operator import itemgetter
import networkx as nx
import random
from networkx.algorithms import community


# In[75]:


from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# In[76]:


# df = pd.read_csv("RFAnet.csv", header=None, names=["source", "target", "rating"], nrows = 10000 ) 
# df.head()

p = 0.10  # 10% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(
        "RFAnet.csv",
        header= None, 
        names=["source", "target", "rating"],
        skiprows=lambda i: i>0 and random.random() > p
)
df.head()


# In[77]:


df = df.drop_duplicates(subset = ['source', 'target'], keep="first")
df.head()


# In[78]:


len(df)


# In[79]:


df["rating_sign"] = df["rating"].apply(lambda x:1 if x > 0 else -1)  #1 means positive edge, -1 means negative edge
df.head()


# In[80]:


df["source"].nunique()


# In[81]:


df["target"].nunique()


# In[82]:


df.isna().any()


# In[83]:


df["rating_sign"].value_counts()


# In[84]:


nodelist = np.unique(df[['source', 'target']].values.flatten())


# In[85]:


nodelist.shape


# In[86]:


edgelist = df[['source', 'target']].to_numpy()
# print((edgelist)))
temp = collections.defaultdict(list)
s = 0
for edge in edgelist:
  temp[edge[0]].append(edge[1])
for i in temp.keys():
  s = s + len(temp[i])
print(s, edgelist.shape)


# In[87]:


G = nx.Graph()
for ind in df.index:
  G.add_edge(df['source'][ind], df['target'][ind], rating = df['rating_sign'][ind])


# In[88]:


df[(df['source'] == 3901) & df['target'] == 4779].values


# In[89]:


df.loc[(df['source'] == 3901) & (df['target'] == 4779), 'rating']


# In[90]:


def shortest_path_length(node1, node2):
  try:
    r = G[node1][node2].get('rating')
  except KeyError:
    r = 1
#   r = G[node1][node2].get('rating')
  G.remove_edge(node1,node2)
  if (nx.algorithms.shortest_paths.generic.has_path(G, node1, node2) is False):
    G.add_edge(node1,node2,rating=r)
    return 0
  path_len = nx.shortest_path_length(G, source=node1, target=node2)
  G.add_edge(node1,node2,rating=r)
  return path_len


# In[91]:


def high_degree_nodes(node1, node2):
  try:
    r = G[node1][node2].get('rating')
  except KeyError:
    r = 1
  G.remove_edge(node1,node2)
  if (nx.algorithms.shortest_paths.generic.has_path(G, node1, node2) is False):
    G.add_edge(node1,node2,rating=r)
    return 0
  path = nx.shortest_path(G, node1, node2)
  count = 0
  for node in path:
    if G.degree[node] > 25:
      count += 1
  G.add_edge(node1,node2,rating=r)
  return count


# In[92]:


def ratio_balanced_path(node1, node2):
  try:
    r = G[node1][node2].get('rating')
  except KeyError:
    r = 1
  G.remove_edge(node1,node2)
  if (nx.algorithms.shortest_paths.generic.has_path(G, node1, node2) is False):
    G.add_edge(node1,node2,rating=r)
    return 0
  paths = nx.algorithms.simple_paths.shortest_simple_paths(G, source=node1, target=node2)
  no_paths = 0
  bal_paths = 0
#   count = 0
  for path in paths:
    if len(path) > 4:
      break
#     count += 1
#     if (count > 5):
#         break
    no_paths += 1
    neg_edges = 0
    for src, tgt in pairwise(path):
#       src, tgt = edge
      try:
        if (G[src][tgt].get('rating') == -1):
          neg_edges += 1
      except KeyError:
        continue
#       if (G[src][tgt].get('rating') == 2):  #i.e, if rating is negative
#         neg_edges += 1
    if (neg_edges%2 == 0):
      bal_paths += 1
  G.add_edge(node1,node2,rating=r)
  if (no_paths == 0):
    return 0
  return float(bal_paths/no_paths)


# In[93]:


def ratio_mutual_nodes(node1, node2):
  mut_nodes = 0
  same_sign_nodes = 0
  for node in nx.common_neighbors(G, node1, node2):
    mut_nodes += 1
    try:
      if (G[node1][node].get('rating') == G[node2][node].get('rating')):
        same_sign_nodes += 1
    except KeyError:
      continue
#     if (G[node1][node2].get('rating') == G[node2][node1]["rating"]):
#       same_sign_nodes += 1
#   denom = G.degree(node1) + G.degree(node2) - mut_nodes
#   if denom == 0:
#     return 0
  if mut_nodes == 0:
    return 0
  return float(same_sign_nodes/mut_nodes)


# In[94]:


def propensity(node):
  pos_edges = 0
  edges = 0
  for n in nx.neighbors(G, node):
    edges += 1
    try:
      if (G[node][n].get('rating') == 1):
        pos_edges += 1
    except KeyError:
      continue
#     if (G[node][n].get('rating') == 1):
#       pos_edges += 1
    
  if (edges == 0):
    return 0
  return float(pos_edges/edges)


# In[95]:


centrality_dict = nx.algorithms.centrality.degree_centrality(G)
clustering_dict = nx.algorithms.cluster.clustering(G)


# In[96]:


cols = ['node_1','node_2', 'rating', 'shortest_path_len','high_deg_nodes','ratio_bal_path','ratio_mutual_nodes','node_centrality1','node_centrality2','cluster_coeff1','cluster_coeff2','node_propensity1', 'node_propensity2']

ds = pd.DataFrame(columns = cols)


# In[97]:


for edge in tqdm(G.edges()):
  ds = ds.append({'node_1':edge[0],
                  'node_2':edge[1],
                  'node_centrality1' : centrality_dict[edge[0]],
                  'node_centrality2' : centrality_dict[edge[1]],
                  'node_propensity1': propensity(edge[0]),
                  'node_propensity2': propensity(edge[1]), 
                  'cluster_coeff1' : clustering_dict[edge[0]],
                  'cluster_coeff2' : clustering_dict[edge[1]],
                  'shortest_path_len' : shortest_path_length(edge[0],edge[1]),
                  'high_deg_nodes' : high_degree_nodes(edge[0],edge[1]),
                  'ratio_bal_path': ratio_balanced_path(edge[0],edge[1]),
                  'ratio_mutual_nodes': ratio_mutual_nodes(edge[0],edge[1]),
                  'rating' : G[edge[0]][edge[1]]['rating']
                  }, ignore_index=True )


# In[98]:


n = ratio_balanced_path(5989,8724 )
print(n)


# In[99]:


print(G[5989][7967].get('rating'))


# In[100]:


print(shortest_path_length(5989,8724))


# In[101]:


# ds.drop(columns=['ratio_mutual_nodes'], inplace=True)
# ds.head()


# In[117]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(ds[["shortest_path_len", "high_deg_nodes", "ratio_bal_path"]],ds["rating"],test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(ds[['shortest_path_len','high_deg_nodes','ratio_bal_path','node_centrality1','node_centrality2','cluster_coeff1','ratio_mutual_nodes','cluster_coeff2','node_propensity1', 'node_propensity2']], ds['rating'], test_size=0.3)


# In[118]:


from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model2 = GaussianNB()


# In[119]:


model1.fit(X_train1,y_train1)
model2.fit(X_train2,y_train2)


# In[120]:


model1.score(X_test1,y_test1)


# In[121]:


model2.score(X_test2,y_test2)


# In[122]:


get_ipython().system('jt -t onedork')


# In[ ]:




