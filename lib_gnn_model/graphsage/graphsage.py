import os
import logging

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler

from lib_gnn_model.graphsage.graphsage_net import GraphSageNet
from lib_gnn_model.gnn_base import GNNBase
import config


class GraphSAGE(GNNBase):
    def __init__(self, data, num_feats, num_classes):
        super(GraphSAGE, self).__init__()
        
        self.logger = logging.getLogger(__name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = data
        self.x = data.x.to(self.device)
        self.y = data.y.squeeze().to(self.device)
        self.model = GraphSageNet(num_feats, 256, num_classes).to(self.device)

    def train_model(self, train_loader, subgraph_loader, num_epoch=100):
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))
            
            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]
        
                optimizer.zero_grad()
                out = self.model(self.x[n_id], adjs)
                loss = F.nll_loss(out, self.y[n_id[:batch_size]])
                loss.backward()
                optimizer.step()
        
                # total_correct += int(out.argmax(dim=-1).eq(self.y[n_id[:batch_size]]).sum())
        
            train_acc, val_acc, test_acc = self.evaluate_model(subgraph_loader)
            self.logger.info(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    @torch.no_grad()
    def evaluate_model(self, subgraph_loader):
        self.model.eval()
    
        out = self.model.inference(self.x, subgraph_loader, self.device)
    
        y_true = self.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)
    
        results = []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    
        return results
    
    
if __name__ == '__main__':
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    
    dataset_name = 'Cora'
    # dataset = Reddit(config.ORIGINAL_DATASET_PATH + dataset_name)
    dataset = Planetoid(config.ORIGINAL_DATASET_PATH + dataset_name, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=32, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)
    
    graphsage = GraphSAGE(data, dataset.num_features, dataset.num_classes)
    graphsage.train_model(train_loader, subgraph_loader)
    graphsage.evaluate_model()
