import logging
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset

from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.diffpool.diffpool_net import DiffPoolNet
import config


class DiffPool(GNNBase):
    def __init__(self, feat_dim, num_classes, max_nodes, args):
        super(DiffPool, self).__init__(args)

        self.logger = logging.getLogger(__name__)
        self.model = DiffPoolNet(feat_dim, num_classes, max_nodes)

    def train_model(self, train_loader, test_loader, num_epochs=100):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_all = 0

        for epoch in range(num_epochs):
            self.logger.debug('epoch %s' % (epoch,))

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                output, link_loss, entropy_loss = self.model(data.x, data.adj, data.mask)
                loss = F.nll_loss(output, data.y.view(-1))
                loss.backward()
                loss_all += data.y.size(0) * loss.item()
                optimizer.step()

            test_acc = self.evaluate_model(test_loader)
            self.logger.debug('test acc: %s' % (test_acc,))
            self.embedding_dim = self.model.graph_embedding.shape[1]

    @torch.no_grad()
    def evaluate_model(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        for data in test_loader:
            data = data.to(self.device)
            pred = self.model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()

        return correct / len(test_loader.dataset)
