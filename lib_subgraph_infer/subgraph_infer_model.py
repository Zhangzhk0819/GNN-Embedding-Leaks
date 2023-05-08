import logging
import copy

import torch
from sklearn.metrics import roc_auc_score

from lib_gnn_model.gnn_base import GNNBase
from lib_subgraph_infer.subgraph_infer_net import SubgraphInferNet


class SubgraphInferModel(GNNBase):
    def __init__(self, feat_dim, embedding_dim, num_classes, max_nodes, feat_gen_method, args):
        super(SubgraphInferModel, self).__init__(args)

        self.logger = logging.getLogger('subgraph_infer_model')
        self.model = SubgraphInferNet(feat_dim, embedding_dim, num_classes, max_nodes, feat_gen_method, args)

    def train_model(self, train_loader, test_loader, num_epochs=100):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        train_acc_best = 0.0
        state_dict = None
        # torch.autograd.set_detect_anomaly(True)

        # best_val_acc = test_acc = 0
        for epoch in range(num_epochs):
            self.logger.info('epoch %s' % (epoch,))

            for i, (x, adj, mask, graph_embedding, labels) in enumerate(train_loader):
                x = x.to(self.device)
                adj = adj.to(self.device)
                mask = mask.to(self.device)
                graph_embedding = graph_embedding.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(x, adj, mask, graph_embedding)

                # if torch.any(output.isnan()):
                #     continue

                loss = self.model.loss(output, labels.view(-1))
                # with torch.autograd.detect_anomaly():
                loss.backward()
                optimizer.step()

            train_acc = self.evaluate_model(train_loader)
            test_acc = self.evaluate_model(test_loader)
            # self.logger.info('test acc: %s' % (test_acc,))
            self.logger.info('train acc: %s, test acc: %s' % (train_acc, test_acc))
            # self.embedding_dim = self.model.graph_embedding.shape[1]
            if train_acc > train_acc_best:
                train_acc_best = train_acc
                state_dict = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate_model(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        for x, adj, mask, graph_embedding, labels in test_loader:
            x = x.to(self.device)
            adj = adj.to(self.device)
            mask = mask.to(self.device)
            graph_embedding = graph_embedding.to(self.device)
            labels = labels.to(self.device)

            posterior = self.model(x, adj, mask, graph_embedding)
            pred = torch.max(posterior, dim=1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()

        return correct / len(test_loader.dataset)

    @torch.no_grad()
    def calculate_auc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        posterior_list, label_list = [], []

        for x, adj, mask, graph_embedding, label in test_loader:
            x = x.to(self.device)
            adj = adj.to(self.device)
            mask = mask.to(self.device)
            graph_embedding = graph_embedding.to(self.device)

            posterior_list.append(self.model(x, adj, mask, graph_embedding))
            label_list.append(label)

        posteriors = torch.cat(posterior_list).detach().cpu().numpy()
        labels = torch.cat(label_list).detach().cpu().numpy()

        return roc_auc_score(labels, posteriors[:, 1])
