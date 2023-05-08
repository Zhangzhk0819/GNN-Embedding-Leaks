import logging
import os.path as osp
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.data import DenseDataLoader

import config
from lib_dataset.tu_dataset import TUDataset
from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.mincut_pool.mincut_pool_net import MinCutPoolNet


class MinCutPool(GNNBase):
    def __init__(self, num_feats, num_classes, max_nodes, args):
        super(MinCutPool, self).__init__(args)
        
        self.logger = logging.getLogger(__name__)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MinCutPoolNet(num_feats, num_classes, max_nodes)

    def train_model(self, train_loader, test_loader, num_epoch=100):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out, mc_loss, o_loss = self.model(data.x, data.adj, data.mask)
                # out, mc_loss, o_loss = self.model(data.x, data.edge_index, data.batch)
                # loss = F.nll_loss(out, data.y.view(-1)) + mc_loss + o_loss
                loss = F.nll_loss(out, data.y.view(-1))
                loss.backward()
                optimizer.step()

            train_acc = self.evaluate_model(test_loader)
            self.logger.info('train acc: %s' % (train_acc,))
            self.embedding_dim = self.model.graph_embedding.shape[1]

    @torch.no_grad()
    def evaluate_model(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0
    
        for data in test_loader:
            data = data.to(self.device)
            # pred, mc_loss, o_loss = self.model(data.x, data.edge_index, data.batch)
            pred, mc_loss, o_loss = self.model(data.x, data.adj, data.mask)
            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
    
        return correct / len(test_loader.dataset)

