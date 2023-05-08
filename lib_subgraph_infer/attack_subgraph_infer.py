import logging
import pickle

import numpy as np
from torch.utils.data import DataLoader

from lib_subgraph_infer.attack import Attack
from lib_subgraph_infer.subgraph_infer_model import SubgraphInferModel
from lib_subgraph_infer.subgraph_dataset import SubgraphDataset


class AttackSubgraphInfer(Attack):
    def __init__(self, target_model, shadow_model, embedding_dim, num_classes, args):
        super(AttackSubgraphInfer, self).__init__(target_model, shadow_model, embedding_dim, num_classes, args)

        self.logger = logging.getLogger('attack_subgraph_infer_2')

    def generate_train_data(self, pos_dataset, neg_dataset):
        self.logger.info('generating train data')

        self.train_graph_embedding = np.zeros([len(pos_dataset), self.embedding_dim])
        self.train_positive_subgraph = []

        for i, data in enumerate(pos_dataset):
            self.logger.debug('generating %s training data' % (i,))

            x, adj, mask = self.generate_input_data(data)
            self.shadow_model(x, adj, mask)

            self.train_graph_embedding[i] = self.shadow_model.graph_embedding.cpu().detach().numpy()
            self.train_positive_subgraph.append(self._gen_subgraph_data(data))

        self.train_negative_subgraph = self._gen_negative_subgraph(neg_dataset, len(pos_dataset))

    def generate_test_data(self, pos_dataset, neg_dataset):
        self.logger.info('generating test data')

        self.test_graph_embedding = np.zeros([len(pos_dataset), self.embedding_dim])
        self.test_positive_subgraph = []

        for i, data in enumerate(pos_dataset):
            self.logger.debug('generating %s testing data' % (i,))

            x, adj, mask = self.generate_input_data(data)
            self.target_model(x, adj, mask)

            self.test_graph_embedding[i] = self.target_model.graph_embedding.cpu().detach().numpy()
            self.test_positive_subgraph.append(self._gen_subgraph_data(data))

        self.test_negative_subgraph = self._gen_negative_subgraph(neg_dataset, len(pos_dataset))

    def generate_dataloader(self):
        self.logger.info('generating dataloader')

        self.train_dataloader = self._gen_dataloader(self.train_graph_embedding, self.train_positive_subgraph, self.train_negative_subgraph)
        self.test_dataloader = self._gen_dataloader(self.test_graph_embedding, self.test_positive_subgraph, self.test_negative_subgraph, shuffle=False)

    def train_attack_model(self, feat_dim, feat_gen_method, is_train=True):
        self.logger.info('training attack model')

        self.attack_model = SubgraphInferModel(feat_dim, self.embedding_dim, self.num_classes, self.args['max_nodes'], feat_gen_method, self.args)

        if is_train:
            self.attack_model.train_model(self.train_dataloader, self.test_dataloader, num_epochs=100)

    def evaluate_attack_model(self):
        self.logger.info('evaluating attack model')

        acc = self.attack_model.evaluate_model(self.test_dataloader)
        auc = self.attack_model.calculate_auc(self.test_dataloader)

        return acc, auc

    def save_data(self, save_path):
        save_data = {
            'train_graph_embedding': self.train_graph_embedding,
            'train_positive_subgraph': self.train_positive_subgraph,
            'train_negative_subgraph': self.train_negative_subgraph,
            'test_graph_embedding': self.test_graph_embedding,
            'test_positive_subgraph': self.test_positive_subgraph,
            'test_negative_subgraph': self.test_negative_subgraph
        }
        pickle.dump(save_data, open(save_path, 'wb'))

    def load_data(self, save_path):
        load_data = pickle.load(open(save_path, 'rb'))
        self.train_graph_embedding = load_data['train_graph_embedding']
        self.train_positive_subgraph = load_data['train_positive_subgraph']
        self.train_negative_subgraph = load_data['train_negative_subgraph']
        self.test_graph_embedding = load_data['test_graph_embedding']
        self.test_positive_subgraph = load_data['test_positive_subgraph']
        self.test_negative_subgraph = load_data['test_negative_subgraph']

    def save_attack_model(self, save_path):
        self.attack_model.save_model(save_path)

    def load_attack_model(self, save_path):
        self.attack_model.load_model(save_path)

    def _gen_subgraph_data(self, graph):
        subgraph = self.generate_subgraph(graph)
        subgraph_nodes = subgraph.nodes

        x, adj, mask = self.generate_subgraph_data(graph, subgraph_nodes)
        ret_subgraph_data = {'x': x, 'adj': adj, 'mask': mask}

        return ret_subgraph_data

    def _gen_negative_subgraph(self, graphs, num_subgraph):
        ret_subgraphs = []

        for i in range(num_subgraph):
            self.logger.debug('generating %s negative subgraphs' % (i,))

            graph_index = np.random.choice(np.arange(len(graphs)))
            graph = graphs[int(graph_index)]
            ret_subgraphs.append(self._gen_subgraph_data(graph))

        return ret_subgraphs

    def _gen_dataloader(self, graph_embedding, positive_subgraph, negative_subgraph, shuffle=True):
        dataset = SubgraphDataset(graph_embedding, positive_subgraph, negative_subgraph)
        return DataLoader(dataset=dataset, batch_size=32, shuffle=shuffle)
