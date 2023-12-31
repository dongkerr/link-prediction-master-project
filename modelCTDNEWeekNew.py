# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12n2c14DTzGRf2TlgAn631gv9Wmqy7yUC
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
# if 'google.colab' in sys.modules:
#    %pip install -q stellargraph[demos]==1.2.1

import sys
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display, HTML
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from stellargraph import StellarGraph
from stellargraph.datasets import IAEnronEmployees
from datetime import datetime
import time
# %matplotlib inline

def buildEdge(graph):
    xList = []
    yList = []
    dictList = {}
    weightList = []

    for x, y, dic in graph.edges(data=True):
        # print(x,y,weight,dic)

        xList.append(x)
        yList.append(y)
        for key in dic.keys():
            if key not in dictList:
                dictList[key] = [dic[key]]
            else:
                dictList[key].append(dic[key])
        # j+=1
        # if j==10:
        #   break
    edgesList=pd.DataFrame(
        {
            'source':xList,
            'target':yList,
            'value':dictList['value'],
            'n_tx':dictList['n_tx'],
        }
    )
    return edgesList
pathCommen = "/home/user/weizi/data/graphCommen2020-10-31week.graphml"
pathUncommen = "/home/user/weizi/data/graphUncommen2020-10-31week.graphml"

#reading data to build graph

graphCommen = nx.read_graphml(pathCommen)
edgesComenList = buildEdge(graphCommen)
graphUncommen = nx.read_graphml(pathUncommen)
edgesUncomenList = buildEdge(graphUncommen)

print('len(edgesList):',len(edgesComenList),len(edgesUncomenList))

from stellargraph import StellarDiGraph
def buildStellarGraph(edgesList):
  stellarGraph_features = StellarDiGraph(
      edges=edgesList,
      node_type_default="user",
      edge_type_default="send",
      edge_weight_column="weight"
      # {"user": list(G.nodes())}, {"send": edgesList}
  )
  return stellarGraph_features
stellarGraphCommen = buildStellarGraph(edgesComenList)
stellarGraphUncommen = buildStellarGraph(edgesUncomenList)
print(stellarGraphCommen.info(),stellarGraphUncommen.info())



#Create link examples for training and testing
# subset of edges to split
train_subset = 0.75
test_subset = 0.25

# number of edges to be kept in the graph
num_edges_train = int(len(edgesComenList) * train_subset)
num_edges_test = len(edgesComenList) - num_edges_train
# keep older edges in graph, and predict more recent edges
edges_graph = edgesComenList+edgesUncomenList

print("num_edges_train,test",num_edges_train,num_edges_test)
# split recent edges further to train and test sets
# edges_train
# edges_test = train_test_split(edges_other, test_size=test_subset)

print(
    f"Number of edges in graph: {len(edges_graph)}\n"
    # f"Number of edges in training set: {len(edges_train)}\n"
    # f"Number of edges in test set: {len(edges_test)}"
)

graph = StellarDiGraph(
    edges=edges_graph,
    node_type_default="user",
     edge_type_default="send",
     edge_weight_column="weight"
    # {"user": list(G.nodes())}, {"send": edgesList}
)
print("all graph",graph.info())

def positive_and_negative_links(g, edges):
    pos = list(edges[["source", "target"]].itertuples(index=False))
    neg = sample_negative_examples(g, pos)
    return pos, neg


def sample_negative_examples(g, positive_examples):
    positive_set = set(positive_examples)
    negative_set = set([])
    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt
            and
            # neither direction of the edge should be a positive one
            # print((src,tgt),'\n',positive_set,'\n',negative_set)
            (src, tgt) not in positive_set
            and (tgt, src) not in positive_set
            and (src, tgt) not in negative_set
            and (tgt, src) not in negative_set
        )
    num = 0
    possible_neg_edges = []
    print(len(positive_examples))
    while num<=len(positive_examples):
      src = random.sample(list(graph.nodes()),1)[0]
      tgt = random.sample(list(graph.nodes()),1)[0]
      print('src and tgt',src,tgt)
      if valid_neg_edge(src,tgt):
        negative_set.add((src,tgt))
        possible_neg_edges.append((src,tgt))
        num+=1
        print(num)
    return possible_neg_edges

# print(type(graph.nodes()),graph.nodes(),graph.nodes()[random.randint(0, len(graph.nodes())-1)])

# pos, neg = positive_and_negative_links(graph, edges_train)
# len_train = int(len(edges_otherPositive) * (1 - test_subset))
# pos_train =  list(edges_otherPositive[:len_train][["source", "target"]].itertuples(index=False))
# neg_train = list(edges_otherNegative[:len_train][["source", "target"]].itertuples(index=False))
# # pos_test, neg_test = positive_and_negative_links(graph, edges_test)
# pos_test =  list(edges_otherPositive[len_train:][["source", "target"]].itertuples(index=False))
# neg_test = list(edges_otherNegative[len_train:len(edges_otherPositive)][["source", "target"]].itertuples(index=False))
# edgesComenList[:num_edges_grapC]+edgesUncomenList[:num_edges_grapUC]
pos_train =  list(edgesComenList[:num_edges_train][["source", "target"]].itertuples(index=False))
neg_train = list(edgesUncomenList[:num_edges_train][["source", "target"]].itertuples(index=False))
# pos_test, neg_test = positive_and_negative_links(graph, edges_test)
pos_test =  list(edgesComenList[num_edges_train:][["source", "target"]].itertuples(index=False))
neg_test = list(edgesUncomenList[num_edges_train:num_edges_train+num_edges_test][["source", "target"]].itertuples(index=False))


print("train edge len",len(pos_train),len(neg_train),"test edge len",len(pos_test),len(neg_test))
# print(
#     f"{graph.info()}\n"
#     f"Training examples: {len(pos)} positive links, {len(neg)} negative links\n"
#     f"Test examples: {len(pos_test)} positive links, {len(neg_test)} negative links"
# )

num_walks_per_node = 1
walk_length = 80
context_window_size = 2

countEmbedding1 = 0
countEmbedding0 = 0
num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1)

from stellargraph.data import TemporalRandomWalk

temporal_rw = TemporalRandomWalk(graph)
temporal_walks = temporal_rw.run(
    num_cw=num_cw,
    cw_size=context_window_size,
    max_walk_length=walk_length,
    walk_bias="exponential",
)

print("Number of temporal random walks: {}".format(len(temporal_walks)))

from gensim.models import Word2Vec

embedding_size = 128
temporal_model = Word2Vec(
    temporal_walks,
    vector_size = embedding_size,
    window=context_window_size,
    min_count=0,
    sg=1,
    workers=2,
    epochs=1,
)

unseen_node_embedding = np.zeros(embedding_size)

def temporal_embedding(u):
    try:
        # countEmbedding1 += 1
        return temporal_model.wv[u]
    except KeyError:
        # countEmbedding0 += 1
        return unseen_node_embedding

def plot_tsne(title, x, y=None):
    tsne = TSNE(n_components=2)
    x_t = tsne.fit_transform(x)

    plt.figure(figsize=(7, 7))
    plt.title(title)
    alpha = 0.7 if y is None else 0.5

    scatter = plt.scatter(x_t[:, 0], x_t[:, 1], c=y, cmap="jet", alpha=alpha)
    if y is not None:
        plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")

# temporal_node_embeddings = temporal_model.wv.vectors
# plot_tsne("TSNE visualisation of temporal node embeddings", temporal_node_embeddings)

def operator_l2(u, v):
    return (u - v) ** 2


binary_operator = operator_l2

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV,BayesianRidge,LassoLars
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.preprocessing import StandardScaler


def link_examples_to_features(link_examples, transform_node):
    op_func = (
        operator_func[binary_operator]
        if isinstance(binary_operator, str)
        else binary_operator
    )
    return [
        op_func(transform_node(src), transform_node(dst)) for src, dst in link_examples
    ]


def link_prediction_classifier(max_iter=2000):
    # lr_clf = BayesianRidge()
    lr_clf = LassoLars(alpha=.1)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict(link_features)

    # check which class corresponds to positive links
    # positive_column = list(clf.classes_).index(1)
    # return roc_auc_score(link_labels, predicted[:, positive_column])
    mse = mean_squared_error(link_labels, predicted)
    print("MSE: %.2f" % mse)

def labelled_links(positive_examples, negative_examples):
    example = positive_examples + negative_examples
    label = np.repeat([1, 0], [len(positive_examples), len(negative_examples)])
    combined = list(zip(example, label))
        
    random.shuffle(combined)

    a, b = zip(*combined)
    return (a,b)


link_examples, link_labels = labelled_links(pos_train, neg_train)
# link_examples_test, link_labels_test = labelled_links(pos_test, neg_test)

#for test
# fileTest="/home/user/weizi/data/2020-12-14.graphml"
# graphTest = nx.read_graphml(fileTest)
# dayTest = 12
# dayLenTest = 1
# yearTest = '2020'
# monthTest = '12'
# edgesTest = buildEdge(dayTest,dayLenTest,yearTest,monthTest,[graphTest])
# graphTest = buildStellarGraph(edgesTest)
# print(graphTest.info())

# posTest, negTest = positive_and_negative_links(graphTest, edgesTest)
# graphTest = graph
link_examples_Test, link_labels_Test = labelled_links(pos_test, neg_test)
# print(link_examples_Test,'\n', len(link_labels_Test),link_labels_Test)

temporal_clf_Test = link_prediction_classifier()
temporal_link_features = link_examples_to_features(link_examples, temporal_embedding)
temporal_link_features_test = link_examples_to_features(
    link_examples_Test, temporal_embedding
)
# temporal_clf_Test.fit(temporal_link_features, link_labels)
temporal_clf_Test.fit(temporal_link_features, link_labels)
temporal_score_Test = evaluate_roc_auc(
    temporal_clf_Test, temporal_link_features_test, link_labels_Test
)
# print(f"Score (ROC AUC): {temporal_score_Test:.2f}")
print("embedding:",countEmbedding1,countEmbedding0)

# print(len(link_examples),'\n',link_examples_test,'\n', link_labels_test)

# plot_tsne("temporal link embeddings", temporal_link_features_test_Test, link_labels_test)
#
# print(type(temporal_link_features_test_Test), type(link_labels_test))







