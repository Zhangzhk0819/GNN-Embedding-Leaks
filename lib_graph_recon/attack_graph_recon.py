import torch
from torch_geometric.data import DenseDataLoader
from torch.utils.data.dataloader import DataLoader
import numpy as np

from lib_gnn_model.graphvae.graphvae import GraphVAE
from lib_graph_recon.attack import Attack
from lib_graph_recon.fine_tune_dataset import FineTuneDataset


class AttackGraphRecon(Attack):
    def __init__(self, target_model, max_nodes, args):
        super(AttackGraphRecon, self).__init__(target_model, max_nodes, args)

    def init_graph_vae(self, dataset, embedding_dim, max_nodes):
        self.graph_vae = GraphVAE(dataset.num_features, embedding_dim,
                                  dataset.num_classes, max_nodes, self.args)

    def train_gae(self, train_dataset, num_epoch):
        self.logger.info('training gae model')
    
        train_loader = DenseDataLoader(train_dataset, batch_size=1)
        self.graph_vae.train_model(train_loader, num_epoch=num_epoch)
        
    def gen_fine_tune_dataset(self, train_dataset, embedding_dim):
        self.logger.info('generating embedding')
        graph_embedding = torch.zeros([len(train_dataset), embedding_dim], dtype=torch.float32)
    
        for i, data in enumerate(train_dataset):
            x, adj, mask = self._gen_input_data(data)
            self.target_model(x, adj, mask)
            graph_embedding[i] = self.target_model.graph_embedding.cpu().detach().float()

        self.fine_tune_dataset = FineTuneDataset(train_dataset, graph_embedding)
        
    def fine_tune_gae(self, num_epoch):
        self.logger.info('fine tuning gae')
        
        fine_tune_loader = DataLoader(self.fine_tune_dataset, batch_size=1)
        self.graph_vae.fine_tune_model(fine_tune_loader, num_epoch=num_epoch)

    def gen_test_embedding(self, test_dataset):
        self.logger.info('generating test embedding')

        test_graph_embedding = []
        for data in test_dataset:
            x, adj, mask = self._gen_input_data(data)
            self.target_model(x, adj, mask)
            test_graph_embedding.append(self.target_model.graph_embedding)

        self.test_graph_embedding = torch.stack(test_graph_embedding)

    def reconstruct_graph(self):
        self.logger.info('reconstructing graph')
    
        self.recon_adjs = []
        for graph_embedding in self.test_graph_embedding:
            self.recon_adjs.append(self.graph_vae.model.reconstruct(graph_embedding))

    def save_model(self, save_path):
        self.graph_vae.save_model(save_path)

    def load_model(self, save_path):
        self.graph_vae.load_model(save_path)

    def save_data(self, save_path):
        torch.save(self.recon_adjs, save_path)

    def load_data(self, save_path):
        self.recon_adjs = torch.load(save_path)

    def _gen_input_data(self, graph):
        x = graph.x.reshape([1, graph.x.shape[0], graph.x.shape[1]]).to(self.device)
        adj = graph.adj.reshape([1, graph.adj.shape[0], graph.adj.shape[1]]).to(self.device)
        mask = graph.mask.reshape([1, graph.mask.shape[0]]).to(self.device)

        return x, adj, mask
