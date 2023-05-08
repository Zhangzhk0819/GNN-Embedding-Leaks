import torch
import torch.nn.functional as F
from torch.nn import Linear

from lib_gnn_model.diffpool.diffpool_net import GNN


class MeanPoolNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels=64):
        super(MeanPoolNet, self).__init__()
        self.conv = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        
        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv(x, adj, mask))
        
        self.graph_embedding = x.mean(dim=1)
        
        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)
