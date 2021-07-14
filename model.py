import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, n_features, n_classes, k=8, f=8):
        super(GAT, self).__init__()
        self.initial_layer = AttentionLayer(n_in_features=n_features, n_out_features_per_head=f, n_heads=k)
        self.activation_layer = AttentionLayer(n_in_features=k * f, n_out_features_per_head=n_classes, n_heads=k)
        self.k = k
        self.f = f
        self.n_classes = n_classes

    def forward(self, graph, node_features):
        x = self.initial_layer(graph, node_features)
        x = F.elu(x).view(-1, self.k * self.f)
        x = self.activation_layer(graph, x)
        x = torch.mean(x, -2)
        x = x.softmax(-1)
        return x.squeeze(0)


class AttentionLayer(nn.Module):
    def __init__(self, n_in_features, n_out_features_per_head, n_heads):
        super(AttentionLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, n_in_features, n_out_features_per_head)
        self.n_out_features_per_head = n_out_features_per_head
        self.n_heads = n_heads

    def forward(self, graph, node_features):
        x = None
        for i in range(len(graph)):
            h_i = node_features[i].view(1, 1, -1)
            h_js = torch.index_select(node_features, 0, graph[i].neighbors).unsqueeze(0)

            a = self.multi_head_attention(h_i, h_js)

            if x is None:
                x = a
            else:
                x = torch.cat((x, a), 1)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_in_features, n_out_features_per_head):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.n_in_features = n_in_features
        self.n_out_features_per_head = n_out_features_per_head
        self.w = nn.Linear(n_in_features, n_out_features_per_head * n_heads, bias=False)
        self.a = AttentionCoefficient(n_out_features_per_head)

    def forward(self, h_i, h_js):
        # target_features : [BATCH * 1 * IN_FEATURES]
        # neighbor_features : [BATCH * NODES * IN_FEATURES]

        # reshape h_i : [BATCH * 1 * OUT_FEATURES] -> [BATCH * 1 * HEADS * OF_PER_HEAD] -> [BATCH * HEADS * 1 * OF_PER_HEAD]
        h_i = torch.dropout(h_i, p=0.6, train=self.training)
        h_i = self.w(h_i)  # [BATCH * 1 * OUT_FEATURES]
        h_i = h_i.view(-1, 1, self.n_heads, self.n_out_features_per_head)
        h_i = h_i.transpose(-2, -3)

        # reshape h_js : [BATCH * NODES * OUT_FEATURES] -> [BATCH * NODES * HEADS * OF_PER_HEAD] -> [BATCH * HEADS * NODES * OF_PER_HEAD]
        h_js = torch.dropout(h_js, p=0.6, train=self.training)
        h_js = self.w(h_js)
        h_js = h_js.view(-1, h_js.size(1), self.n_heads, self.n_out_features_per_head)
        h_js = h_js.transpose(-2, -3)

        x = self.a(h_i, h_js)  # [BATCH * HEADS * NODES]
        x = torch.softmax(x, -1)  # [BATCH * HEADS * NODES]
        x = x.unsqueeze(-1)  # [BATCH * HEAD * NODES * 1]

        x = x * h_js  # [BATCH * HEAD * NODES * OF_PER_HEAD]

        x = torch.sum(x, -2)  # [BATCH * HEAD * OF_PER_HEAD]
        x = x.view(-1, 1, self.n_heads, self.n_out_features_per_head)  # [BATCH * 1 * HEAD * OF_PER_HEAD]

        return x


class AttentionCoefficient(nn.Module):
    def __init__(self, n_features):
        super(AttentionCoefficient, self).__init__()
        self.a1 = nn.Linear(n_features, 1, bias=False)
        self.a2 = nn.Linear(n_features, 1, bias=False)

    def forward(self, features, neighbors):
        x = self.a1(features) + self.a2(neighbors)  # [BATCH * H * 1 * 1] + [BATCH * H * N * 1] = [BATCH * H * N * 1]
        x = x.squeeze(-1)  # [BATCH * H * N]

        return F.leaky_relu(x)
