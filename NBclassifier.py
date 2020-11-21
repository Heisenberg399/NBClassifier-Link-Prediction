#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from tqdm import tqdm 
import collections
import numpy as np
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community #Importing all the necessory libraries


# In[2]:


from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# In[3]:


df = pd.read_csv("bitdata.csv", header=None, names=["source", "target", "rating", "time"]) #read the DataFrame
df.head()


# In[4]:


df = df.drop_duplicates(subset = ['source', 'target'], keep="first") #removing dupicates entry if any
df.head()


# In[5]:


len(df)


# In[6]:


df["rating_sign"] = df["rating"].apply(lambda x:1 if x > 0 else -1)  #1 means positive edge, -1 means negative edge
df.head()


# In[7]:


df.drop(columns=['time'], inplace=True)
df.head()


# In[8]:


df["source"].nunique() #no of unique members in coulumn source


# In[9]:


df["target"].nunique()


# In[10]:


df.isna().any()


# In[11]:


df["rating_sign"].value_counts() 


# In[12]:


nodelist = np.unique(df[['source', 'target']].values.flatten())


# In[13]:


nodelist.shape


# In[14]:


#creating a list of edges
edgelist = df[['source', 'target']].to_numpy()
# print((edgelist)))
temp = collections.defaultdict(list)
s = 0
for edge in edgelist:
  temp[edge[0]].append(edge[1])
for i in temp.keys():
  s = s + len(temp[i])
print(s, edgelist.shape)


# In[15]:


edgelist


# In[16]:


#creating a graph of the edges 
G = nx.Graph()
for ind in df.index:
  G.add_edge(df['source'][ind], df['target'][ind], rating = df['rating_sign'][ind])


# In[17]:


print(nx.info(G))


# In[18]:


df[(df['source'] == 3901) & df['target'] == 4779].values


# In[19]:


df.loc[(df['source'] == 3901) & (df['target'] == 4779), 'rating']


# In[20]:


#Function for finding the Shortest path between two nodes
def shortest_path_length(node1, node2):
  try:
    r = G[node1][node2].get('rating')
  except KeyError:
    r = 1

  G.remove_edge(node1,node2) #remove edge
  if (nx.algorithms.shortest_paths.generic.has_path(G, node1, node2) is False): 
    G.add_edge(node1,node2,rating=r)
    return 0
  path_len = nx.shortest_path_length(G, source=node1, target=node2)
  G.add_edge(node1,node2,rating=r) #add the shortest egde
  return path_len


# In[21]:


#Function to count high degree nodes
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
    if G.degree[node] > 100: #Nodes with degree >100 will be considered as high degree
      count += 1
  G.add_edge(node1,node2,rating=r)
  return count


# In[22]:


#Function to find ratio of balanced path
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

  for path in paths:
    if len(path) > 4:
      break

    no_paths += 1
    neg_edges = 0
    for src, tgt in pairwise(path):

      try:
        if (G[src][tgt].get('rating') == -1): #rating -1 = negative edge
          neg_edges += 1
      except KeyError:
        continue

    if (neg_edges%2 == 0):
      bal_paths += 1
  G.add_edge(node1,node2,rating=r)
  if (no_paths == 0):
    return 0
  return float(bal_paths/no_paths)


# In[23]:


#Function to find ratio of mutual nodes
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

  if mut_nodes == 0:
    return 0
  return float(same_sign_nodes/mut_nodes)


# In[24]:


#Function to find propensity of node
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

  if (edges == 0):
    return 0
  return float(pos_edges/edges)


# In[25]:


centrality_dict = nx.algorithms.centrality.degree_centrality(G)
clustering_dict = nx.algorithms.cluster.clustering(G)


# In[26]:


#creating a dataset
cols = ['node_1','node_2', 'rating', 'shortest_path_len','high_deg_nodes','ratio_bal_path','ration_mutual_nodes','node_centrality1','node_centrality2','cluster_coeff1','cluster_coeff2','node_propensity1', 'node_propensity2']

ds = pd.DataFrame(columns = cols)


# In[27]:


#inputing values in dataset from the graph
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
                  'ration_mutual_nodes': ratio_mutual_nodes(edge[0],edge[1]),
                  'rating' : G[edge[0]][edge[1]]['rating']
                  }, ignore_index=True )




ds.drop(columns=['ration_mutual_nodes'], inplace=True)
ds.head()


# In[36]:


ds.to_csv("bitcoin-graph.csv", index=None)



#splitting the dataset into training and testing sample
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(ds[["shortest_path_len", "high_deg_nodes", "ratio_bal_path"]],ds["rating"],test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(ds[['shortest_path_len','high_deg_nodes','ratio_bal_path','node_centrality1','node_centrality2','cluster_coeff1','cluster_coeff2','node_propensity1', 'node_propensity2']], ds['rating'], test_size=0.3)


# In[38]:


#applying GaussianNB
from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model2 = GaussianNB()


# In[39]:


model1.fit(X_train1,y_train1) #fits the sample in the model
model2.fit(X_train2,y_train2)


# In[40]:


model1.score(X_test1,y_test1) #calcutates the score of the model


# In[41]:


model2.score(X_test2,y_test2)






from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB()

model1.fit(X_train1,y_train1)

model1.score(X_test1,y_test1)


model1.predict(X_test1)


model1.predict_proba(X_test1)


# In[52]:


#visualising the comarison of NBclassifier with other ML algorithms
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier()


# In[53]:



plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train1, y_train1)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test1)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test1)
        prob_pos =             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value =         calibration_curve(y_test1, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()


# In[ ]:




