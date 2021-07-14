import torch
import pandas as pd
from graph import Graph, Node


def read_cora(path):
    g = Graph()
    contents = pd.read_csv(f'{path}/cora.content', delimiter='\t', header=None)
    cites = pd.read_csv(f'{path}/cora.cites', delimiter='\t')

    indexes = contents.iloc[:, 0].values
    index_dict = {v: i for i, v in enumerate(indexes)}
    features = contents.iloc[:, 1:-1].values
    for idx, i in index_dict.items():
        g.add(Node(i, idx))

    labels = contents.iloc[:, -1]
    label_dict = {v: i for i, v in enumerate(labels.unique())}
    labels = labels.apply(label_dict.get).values

    edges = cites.values
    for n1, n2 in edges:
        g[index_dict[n1]].add_neighbor(g[index_dict[n2]])
        g[index_dict[n2]].add_neighbor(g[index_dict[n1]])

    features, labels = torch.FloatTensor(features), torch.LongTensor(labels)
    return g, features, labels, len(label_dict)


def split_data(g, features, labels, splits):
    s0, s1, s2 = splits[0], splits[0] + splits[1], splits[0] + splits[1] + splits[2]

    train_g, train_features, train_labels = g.split_graph(0, s0), features[:s0], labels[:s0]
    valid_g, valid_features, valid_labels = g.split_graph(s0, s1), features[s0:s1], labels[s0:s1]
    test_g, test_features, test_labels = g.split_graph(s1, s2), features[s1:s2], labels[s1:s2]
    train_g.finalize(), valid_g.finalize(), test_g.finalize()

    return (train_g, train_features, train_labels), \
           (valid_g, valid_features, valid_labels), \
           (test_g, test_features, test_labels)
