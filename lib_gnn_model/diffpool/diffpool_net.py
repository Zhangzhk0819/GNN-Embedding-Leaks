from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class DiffPoolNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, max_nodes, hidden_channels=64):
        super(DiffPoolNet, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_feats, hidden_channels, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(num_feats, hidden_channels, hidden_channels, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        self.graph_embedding = x.mean(dim=1)

        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x
