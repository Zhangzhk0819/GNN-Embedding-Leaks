import logging
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset

import config
from lib_dataset.data_store import DataStore
from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.graphvae.graphvae_net import GraphVAENet


class GraphVAE(GNNBase):
    def __init__(self, feat_dim, embedding_dim, num_classes, max_nodes, args):
        super(GraphVAE, self).__init__(args)
        self.logger = logging.getLogger('graph_vae')

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphVAENet(feat_dim, embedding_dim, num_classes, max_nodes, args)
        self.data_store = DataStore(args, max_nodes)

    def train_model(self, train_loader, num_epoch=10):
        self.model.train()

        optimizer = optim.Adam(list(self.model.parameters()), lr=0.001, weight_decay=0.0001)
        # scheduler = MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))
            loss_all = 0.0
            self.model = self.model.to(self.device)

            for batch_idx, data in enumerate(train_loader):
                self.logger.debug('batch %s' % (batch_idx,))

                data = data.to(self.device)
                
                optimizer.zero_grad()
                loss = self.model(data.x, data.adj, data.mask)
                loss.backward()
                loss_all += loss
            
                optimizer.step()
                scheduler.step()
            
            self.logger.info('training loss %s' % (loss_all,))

            if self.args['is_ablation'] and num_epoch % self.args['epoch_step'] == 0:
                self.data_store.save_graph_vae_model_epoch(self, epoch)
            
    def fine_tune_model(self, fine_tune_loader, num_epoch=10):
        self.model.train()
        # self.model = self.model.to(self.device)
        self.model.encoder = self.model.encoder.to(self.device)
        self.model.decoder = self.model.decoder.to(self.device)

        optimizer = optim.Adam(list(self.model.parameters()), lr=0.001, weight_decay=0.0001)
        # scheduler = MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))
            loss_all = 0.0
        
            for batch_idx, (adj, mask, graph_embedding) in enumerate(fine_tune_loader):
                self.logger.debug('batch %s' % (batch_idx,))
            
                adj = adj.to(self.device)
                mask = mask.to(self.device)
                graph_embedding = graph_embedding.to(self.device)
            
                optimizer.zero_grad()
                loss = self.model.fine_tune(adj, mask, graph_embedding)
                loss.backward()
                loss_all += loss
            
                optimizer.step()
                scheduler.step()
        
            self.logger.info('fine tuning loss %s' % (loss_all,))
                
    @torch.no_grad()
    def evaluate_model(self, test_loader):
        pass
    
    
class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes
    
    
if __name__ == '__main__':
    os.chdir('../../')
    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    
    max_nodes = 20
    # dataset_name = 'PROTEINS'
    dataset_name = 'ENZYMES'
    target_model = 'diff_pool'
    
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
    dataset = TUDataset(config.RAW_DATA_PATH + dataset_name, name=dataset_name, transform=T.ToDense(max_nodes),
                        pre_filter=MyFilter())
    
    dataset = dataset.shuffle()
    train_idx = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]
    test_loader = DenseDataLoader(test_dataset, batch_size=32)
    train_loader = DenseDataLoader(train_dataset, batch_size=1)
    
    graph_vae = GraphVAE(target_model, dataset.num_features, 192, dataset.num_classes, max_nodes)
    graph_vae.train_model(train_loader, 100)
    graph_vae.evaluate_model()
