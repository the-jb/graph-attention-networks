import torch
import torch.nn as nn
import data
from model import GAT


def train(epoch=10000):
    g, features, labels, n_classes = data.read_cora('cora')

    (train_g, train_features, train_labels), (valid_g, valid_features, valid_labels), _ = data.split_data(g, features, labels, (140, 500, 1000))

    model = GAT(n_features=features.shape[1], n_classes=n_classes)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    best_acc = 0
    best_loss = 999
    best_epoch = 0
    for e in range(epoch):
        model.train()
        out = model(train_g, train_features)
        loss = criterion(out, train_labels)
        train_loss = loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(valid_g, valid_features)
            valid_loss = criterion(out, valid_labels).item()
            acc = torch.mean((out.argmax(-1) == valid_labels).float()).item()
        if best_acc < acc or best_loss > valid_loss:
            if best_acc < acc and best_loss > valid_loss:
                torch.save(model.state_dict(), 'cora_model.pt')
            best_acc = max(best_acc, acc)
            best_loss = min(best_loss, valid_loss)
            best_epoch = e
        print(f'epoch {e + 1}: train_loss = {train_loss}, valid_loss = {valid_loss} (best = {best_loss:.4f}) acc = {acc:.4f} (best = {best_acc:.4f})')
        if e > best_epoch + 100:
            break


if __name__ == '__main__':
    train()
