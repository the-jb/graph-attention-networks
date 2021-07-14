import torch
import torch.nn as nn
import data
from model import GAT


def test():
    g, features, labels, n_classes = data.read_cora('cora')
    features, labels = torch.FloatTensor(features), torch.LongTensor(labels)

    _, _, (test_g, test_features, test_labels) = data.split_data(g, features, labels, (140, 500, 1000))
    test_g.finalize()

    model = GAT(n_features=features.shape[1], n_classes=n_classes)
    model.load_state_dict(torch.load('cora_model.pt'))

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        out = model(test_g, test_features)
        loss = criterion(out, test_labels).item()
        acc = torch.mean((out.argmax(-1) == test_labels).float()).item()

    print(f'loss = {loss}, accuracy = {acc}')


if __name__ == '__main__':
    test()
