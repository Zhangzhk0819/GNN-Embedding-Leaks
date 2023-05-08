import logging
import pickle
import random

import numpy as np
import torch
from scipy.special import comb
from torch.utils.data import TensorDataset
import networkx as nx

from lib_classifier.multi_class_classifier import MultiClassClassifier
from utils.convert import to_networkx


class Attack:
    def __init__(self, target_model, shadow_model, args):
        self.logger = logging.getLogger('attack')

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.target_model, self.shadow_model = target_model.to(self.device), shadow_model.to(self.device)
        self.target_model.eval()
        self.shadow_model.eval()

    def determine_attr_fn(self, attr):
        if attr == 'density':
            self.attr_fn = self._density
        elif attr == 'num_nodes':
            self.attr_fn = self._num_nodes
        elif attr == 'num_edges':
            self.attr_fn = self._num_edges
        elif attr == 'diameter':
            self.attr_fn = self._diameter
        elif attr == 'radius':
            self.attr_fn = self._radius
        else:
            raise Exception('unsupported attribute')

    def generate_train_embedding(self, train_dataset, embedding_dim):
        self.logger.info('generating train embedding')

        self.train_graph_embedding = np.zeros([len(train_dataset), embedding_dim])

        for i, data in enumerate(train_dataset):
            x, adj, mask = self._generate_input_data(data)
            self.shadow_model(x, adj, mask)
            self.train_graph_embedding[i] = self.shadow_model.graph_embedding.cpu().detach().numpy()

    def generate_test_embedding(self, test_dataset, embedding_dim):
        self.logger.info('generating test embedding')

        self.test_graph_embedding = np.zeros([len(test_dataset), embedding_dim])

        for i, data in enumerate(test_dataset):
            x, adj, mask = self._generate_input_data(data)
            self.target_model(x, adj, mask)
            self.test_graph_embedding[i] = self.target_model.graph_embedding.cpu().detach().numpy()

    def generate_labels(self, train_dataset, test_dataset, num_class):
        self.logger.info('generating labels')
        self.train_label = np.zeros((len(train_dataset), 1), dtype=np.uint64)
        self.test_label = np.zeros((len(test_dataset), 1), dtype=np.uint64)
        self.baseline_acc = {}

        for property in self.args['properties']:
            self.determine_attr_fn(property)
            train_attr = np.zeros(len(train_dataset))
            test_attr = np.zeros(len(test_dataset))

            for i, graph in enumerate(train_dataset):
                train_attr[i] = self.attr_fn(graph)

            for i, graph in enumerate(test_dataset):
                test_attr[i] = self.attr_fn(graph)

            bins = self._generate_bin(train_attr, num_class)
            train_label = np.digitize(train_attr, bins)
            test_label = np.digitize(test_attr, bins)

            ave_test_label = np.digitize(np.full(test_label.shape[0], np.mean(train_label)), bins)
            self.baseline_acc[property] = np.sum(ave_test_label == test_label) / train_label.size

            self.train_label = np.concatenate((self.train_label, train_label.reshape((-1, 1))), axis=1)
            self.test_label = np.concatenate((self.test_label, test_label.reshape((-1, 1))), axis=1)

        self.train_label = self.train_label[:, 1:]
        self.test_label = self.test_label[:, 1:]

    def train_attack_model(self, is_train=True):
        self.logger.info('training attack model')

        classes_dict = dict(zip(self.args['properties'], [self.args['property_num_class'] for _ in range(len(self.args['properties']))]))
        index_attr_mapping = dict(zip(range(len(self.args['properties'])), self.args['properties']))
        self.attack_model = MultiClassClassifier(self.train_graph_embedding.shape[1], classes_dict, index_attr_mapping)

        if is_train:
            self.attack_model.train_model(self._generate_tensor_dataset(self.train_graph_embedding, self.train_label), num_epochs=100)

    def evaluate_attack_model(self):
        self.logger.info('evaluating attack model')

        test_dset = self._generate_tensor_dataset(self.test_graph_embedding, self.test_label)
        acc = self.attack_model.calculate_multi_class_acc(test_dset)

        return acc

    def save_data(self, save_path):
        embedding_data = {
            'train_embedding': self.train_graph_embedding,
            # 'train_label': self.train_label,
            'test_embedding': self.test_graph_embedding,
            # 'test_label': self.test_label,
        }
        pickle.dump(embedding_data, open(save_path, 'wb'))

    def load_data(self, save_path):
        data = pickle.load(open(save_path, 'rb'))
        self.train_graph_embedding = data['train_embedding']
        # self.train_label = data['train_label']
        self.test_graph_embedding = data['test_embedding']
        # self.test_label = data['test_label']

    def save_attack_model(self, save_path):
        self.attack_model.save_model(save_path)

    def load_attack_model(self, save_path):
        self.attack_model.load_model(save_path)

    def _generate_tensor_dataset(self, feat, label):
        train_x = torch.tensor(np.int64(feat)).float()
        train_y = torch.tensor(np.int64(label))
        return TensorDataset(train_x, train_y)

    def _generate_bin(self, attr, num_class):
        sort_attr = np.sort(attr)
        bins = np.zeros(num_class - 1)
        unit = attr.size / num_class
        for i in range(num_class - 1):
            bins[i] = (sort_attr[int(np.floor(unit * (i + 1)))] + sort_attr[int(np.ceil(unit * (i + 1)))]) / 2

        return bins

    def _density(self, graph):
        num_nodes = graph.num_nodes
        num_edges = np.count_nonzero(graph.adj.numpy()) / 2
        return num_edges / comb(num_nodes, 2)

    def _num_nodes(self, graph):
        return graph.num_nodes

    def _num_edges(self, graph):
        return np.count_nonzero(graph.adj.numpy()) / 2

    def _to_nx_graph(self, graph):
        nx_graph = to_networkx(graph, to_undirected=True)
        if not nx.is_connected(nx_graph):
            self.logger.debug('graph unconnected, generate random edge to connect it')
            self._connect_nx_graph(nx_graph)
        return nx_graph

    def _diameter(self, graph):
        nx_graph = self._to_nx_graph(graph)
        return nx.algorithms.diameter(nx_graph)

    # todo: add other properties mimic diameter, remember to add the corresponding function to determine_attr_fn()

    def _eccentricity(self, graph):
        # The eccentricity of a node v is the maximum distance from v to all other nodes in G.
        nx_graph = self._to_nx_graph(graph)
        return nx.algorithms.eccentricity(nx_graph)

    def _radius(self, graph):
        # The radius is the minimum eccentricity.
        nx_graph = self._to_nx_graph(graph)
        return nx.algorithms.radius(nx_graph)

    def _periphery(self, graph):
        # The periphery is the set of nodes with eccentricity equal to the diameter.
        nx_graph = self._to_nx_graph(graph)
        return nx.algorithms.periphery(nx_graph)

    def _center(self, graph):
        # The center is the set of nodes with eccentricity equal to radius.
        nx_graph = self._to_nx_graph(graph)
        return nx.algorithms.center(nx_graph)

    def _generate_input_data(self, graph):
        x = graph.x.reshape([1, graph.x.shape[0], graph.x.shape[1]]).cuda()
        adj = graph.adj.reshape([1, graph.adj.shape[0], graph.adj.shape[1]]).cuda()
        mask = graph.mask.reshape([1, graph.mask.shape[0]]).cuda()

        return x, adj, mask

    def _connect_nx_graph(self, nx_graph):
        components = list(nx.connected_components(nx_graph))
        pre_component = components[0]

        for component in components[1:]:
            v1 = random.choice(tuple(pre_component))
            v2 = random.choice(tuple(component))
            nx_graph.add_edge(v1, v2)
            pre_component = component


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
