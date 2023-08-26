import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from stellargraph.mapper import KGTripleGenerator
from stellargraph.layer import ComplEx
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model

#access the data
g0 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2019-12-31.graphml.bz2')
for i in range(1, 5):
    globals()["g"+str(i)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-01-0' + str(i) + '.graphml.bz2')
for j in range(0, 3):
    globals()["g"+str(5 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-01-' + str(11 + j*7) + '.graphml.bz2')
g8 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-02-01.graphml.bz2')
g9 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-02-08.graphml.bz2')
for j in range(2, 5):
    globals()["g"+str(8 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-02-' + str(1 + j*7) + '.graphml.bz2')
g13 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-03-07.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(13 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-03-' + str(7 + j*7) + '.graphml.bz2')
g17 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-04-04.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(17 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-04-' + str(4 + j*7) + '.graphml.bz2')
g21 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-05-02.graphml.bz2')
g22 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-05-09.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(22 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-05-' + str(9 + j*7) + '.graphml.bz2')
g26 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-06-06.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(26 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-06-' + str(6 + j*7) + '.graphml.bz2')
g30 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-07-04.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(30 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-07-' + str(4 + j*7) + '.graphml.bz2')
g34 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-08-01.graphml.bz2')
g35 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-08-08.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(35 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-08-' + str(8 + j*7) + '.graphml.bz2')
g39 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-09-05.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(39 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-09-' + str(5 + j*7) + '.graphml.bz2')
g43 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-10-03.graphml.bz2')
for j in range(1, 5):
    globals()["g"+str(43 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-10-' + str(3 + j*7) + '.graphml.bz2')
g48 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-11-07.graphml.bz2')
for j in range(1, 4):
    globals()["g"+str(48 + j)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_week/2020-11-' + str(7 + j*7) + '.graphml.bz2')
g52 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-11-29.graphml.bz2')
g53 = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-11-30.graphml.bz2')
for i in range(1, 10):
    globals()["g"+str(i + 53)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-12-0' + str(i) + '.graphml.bz2')
for i in range(10, 15):
    globals()["g"+str(i + 53)] = nx.read_graphml('/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-12-' + str(i) + '.graphml.bz2')


#removes the nodes not in g0 from g1-g67
for i in range(1, 68):
    globals()["g"+str(i)+"_copy"] = globals()["g"+str(i)].copy()
    globals()["g"+str(i)+"_copy"].remove_nodes_from(n for n in globals()["g"+str(i)] if n not in g0)

#compose g1_copy-g67_copy

R = nx.compose(g1_copy, g2_copy)
attr_n_tx = {e: g1_copy.edges[e]['n_tx'] + g2_copy.edges[e]['n_tx'] for e in g1_copy.edges & g2_copy.edges}
nx.set_edge_attributes(R, attr_n_tx, 'n_tx')
attr_value = {e: g1_copy.edges[e]['value'] + g2_copy.edges[e]['value'] for e in g1_copy.edges & g2_copy.edges}
nx.set_edge_attributes(R, attr_value, 'value')
for i in range(3,68):
    R = nx.compose(R, globals()["g"+str(i)+"_copy"])
    attr_n_tx = {e: R.edges[e]['n_tx'] + globals()["g"+str(i)+"_copy"].edges[e]['n_tx'] for e in R.edges & globals()["g"+str(i)+"_copy"].edges}
    nx.set_edge_attributes(R, attr_n_tx, 'n_tx')
    attr_value = {e: R.edges[e]['value'] + globals()["g"+str(i)+"_copy"].edges[e]['value'] for e in R.edges & globals()["g"+str(i)+"_copy"].edges}
    nx.set_edge_attributes(R, attr_value, 'value')
    i += 1

print('R.number_of_nodes: ' + str(R.number_of_nodes))
print('R.number_of_edges: ' + str(R.number_of_edges()))


R = StellarGraph.from_networkx(R,edge_weight_attr="n_tx" and "value")
print(R.info())

# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(R)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)
#graph_test = StellarGraph.from_networkx(graph_test)
print(graph_test.info())

# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(graph_test, R)
graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global"
)
(
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

print(graph_train.info())

df1 = pd.DataFrame(
    [
        (
            "Training Set",
            len(examples_train),
            "Train Graph",
            "Old Graph",
            "Train the Link Classifier",
        ),
        (
            "Model Selection",
            len(examples_model_selection),
            "Train Graph",
            "Old Graph",
            "Select the best Link Classifier model",
        ),
        (
            "Test set",
            len(examples_test),
            "Test Graph",
            "New Graph",
            "Evaluate the best Link Classifier",
        ),
    ],
    columns=("Split", "Number of Examples", "Hidden from", "Picked from", "Use"),
).set_index("Split")
display(df1)

p = 1.0
q = 1.0
dimensions = 128
num_walks = 1
walk_length = 100
window_size = 10
num_epochs = 1
workers = multiprocessing.cpu_count()

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec


def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_epochs,
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

embedding_train = node2vec_embedding(graph_train, "Train Graph")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


binary_operators = [operator_hadamard, operator_l1, operator_l2]

results = [run_link_prediction(op) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])

print(f"Best result from '{best_result['binary_operator'].__name__}'")

df2=pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")
display(df2)

embedding_test = node2vec_embedding(graph_test, "Test Graph")

test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_test,
    best_result["binary_operator"],
)
print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
)

# Calculate edge features for test data
link_features = link_examples_to_features(
    examples_test, embedding_test, best_result["binary_operator"]
)

# Learn a projection from 128 dimensions to 2
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(link_features)

# plot the 2-dimensional points
plt.figure(figsize=(16, 12))
plt.scatter(
    X_transformed[:, 0],
    X_transformed[:, 1],
    c=np.where(labels_test == 1, "b", "r"),
    alpha=0.5,
)
plt.show()
plt.savefig('WDK_year_traintestg2_v2.png')

