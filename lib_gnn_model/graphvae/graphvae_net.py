import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from lib_gnn_model.diffpool.diffpool_net import DiffPoolNet
from lib_gnn_model.mean_pool.mean_pool_net import MeanPoolNet
from lib_gnn_model.mincut_pool.mincut_pool_net import MinCutPoolNet


class GraphVAENet(nn.Module):
    def __init__(self, feat_dim, embedding_dim, num_classes, max_num_nodes, args):
        super(GraphVAENet, self).__init__()
        
        self.args = args
        
        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_num_nodes = max_num_nodes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.encoder = self.determine_target_model()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 250),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def determine_target_model(self):
        if self.args['encoder_method'] == 'diff_pool':
            target_model = DiffPoolNet(self.feat_dim, self.num_classes, self.max_num_nodes)
        elif self.args['encoder_method'] == 'mincut_pool':
            target_model = MinCutPoolNet(self.feat_dim, self.num_classes, self.max_num_nodes)
        elif self.args['encoder_method'] == 'mean_pool':
            target_model = MeanPoolNet(self.feat_dim, self.num_classes)
        else:
            raise Exception('unsupported target model')
        
        return target_model

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)

        # todo: use matrix operation to speed up this step
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init

        # todo: use matrix operation to speed up this step
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :]) for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            
            norm = torch.norm(x_new)
            x = x_new / norm

        return x

    def deg_feat_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def forward(self, x, adj, mask):
        self.encoder(x, adj, mask)
        recon_adj_vec = self.decoder(self.encoder.graph_embedding)
        recon_adj_vec = torch.sigmoid(recon_adj_vec)

        recon_adj_lower = self.recover_adj_lower(recon_adj_vec.cpu().data)
        recon_adj = self.recover_full_adj_from_lower(recon_adj_lower)

        ################################## matching #######################################
        # the following part can be run in cpu
        # set matching features be degree
        recon_adj_feat = torch.sum(recon_adj, 1)

        ori_adj = adj.cpu().data[0]
        ori_adj[mask.cpu().data[0], mask.cpu().data[0]] = 1
        ori_adj_feat = torch.sum(ori_adj, 1)

        S = self.edge_similarity_matrix(ori_adj, recon_adj, ori_adj_feat, recon_adj_feat, self.deg_feat_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        assignment = self.mpm(init_assignment, S)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        # order row index according to col index
        adj_permuted = self.permute_adj(ori_adj, row_ind, col_ind)
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes)) == 1].squeeze_()
        #########################################################################################

        adj_vectorized = adj_vectorized.to(self.device)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized, recon_adj_vec[0])

        return adj_recon_loss

    def fine_tune(self, adj, mask, graph_embedding):
        recon_adj_vec = self.decoder(graph_embedding)
        recon_adj_vec = torch.sigmoid(recon_adj_vec)
    
        recon_adj_lower = self.recover_adj_lower(recon_adj_vec.cpu().data)
        recon_adj = self.recover_full_adj_from_lower(recon_adj_lower)
    
        ################################## matching #######################################
        # the following part can be run in cpu
        # set matching features be degree
        recon_adj_feat = torch.sum(recon_adj, 1)
    
        ori_adj = adj.cpu().data[0]
        ori_adj[mask.cpu().data[0], mask.cpu().data[0]] = 1
        ori_adj_feat = torch.sum(ori_adj, 1)
    
        S = self.edge_similarity_matrix(ori_adj, recon_adj, ori_adj_feat, recon_adj_feat, self.deg_feat_similarity)
    
        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        assignment = self.mpm(init_assignment, S)
    
        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        # order row index according to col index
        adj_permuted = self.permute_adj(ori_adj, row_ind, col_ind)
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1].squeeze_()
        #########################################################################################
    
        adj_vectorized = adj_vectorized.to(self.device)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized, recon_adj_vec[0])
    
        return adj_recon_loss
    
    def reconstruct(self, graph_embedding):
        self.decoder = self.decoder.to(self.device)
        recon_adj_vec = self.decoder(graph_embedding)
        recon_adj_vec = torch.sigmoid(recon_adj_vec)
        recon_adj_vec = torch.bernoulli(recon_adj_vec).detach().cpu()
        recon_adj_lower = self.recover_adj_lower(recon_adj_vec.data)
        recon_adj = self.recover_full_adj_from_lower(recon_adj_lower)

        return recon_adj

    def adj_recon_loss(self, adj_truth, adj_pred):
        # return F.binary_cross_entropy(adj_truth, adj_pred.detach())
        return F.binary_cross_entropy(adj_pred, adj_truth)
