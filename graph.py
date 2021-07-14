import torch


class Node:
    def __init__(self, idx, origin_idx):
        self.idx = idx
        self.origin_idx = origin_idx
        self._neighbors = {self: 1}
        self._neighbor_idxs = [idx]

    def add_neighbor(self, node):
        self._neighbors[node] = 1
        self._neighbor_idxs.append(node.idx)

    def is_neighbor(self, node):
        return node in self._neighbors

    def finalize(self):
        self._neighbor_idxs = torch.tensor(self._neighbor_idxs, dtype=torch.int64)

    @property
    def neighbors(self):
        return self._neighbor_idxs


class Graph:
    def __init__(self):
        self._nodes = []

    def add(self, node):
        self._nodes.append(node)

    def finalize(self):
        for node in self._nodes:
            node.finalize()

    def split_graph(self, start, end):
        g = Graph()
        split_nodes = self._nodes[start:end]
        for i, node in enumerate(split_nodes):
            n = Node(i, node.origin_idx)
            g.add(n)

        for i, node in enumerate(split_nodes):
            for neighbor_idx in node.neighbors:
                if start <= neighbor_idx < end:
                    g[i].add_neighbor(g[neighbor_idx - start])
                    g[neighbor_idx - start].add_neighbor(g[i])
        return g

    def __contains__(self, item):
        return item in self._nodes

    def __getitem__(self, i):
        return self._nodes[i]

    def __len__(self):
        return len(self._nodes)
