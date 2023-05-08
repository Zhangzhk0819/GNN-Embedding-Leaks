import datetime
import logging
import time
import os

from torch_geometric.data import DenseDataLoader
import numpy as np

import config
from lib_dataset.data_store import DataStore
from lib_gnn_model.diffpool.diffpool import DiffPool
from lib_gnn_model.mincut_pool.mincut_pool import MinCutPool
from lib_gnn_model.mean_pool.mean_pool import MeanPool


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger(__name__)

        self.args = args
        self.start_time = datetime.datetime.now()
        self.dataset_name = args['dataset_name']

        self.determine_max_nodes()
        self.data_store = DataStore(args, self.max_nodes)
        self.load_data()

        self.split_data()
        self.gen_attack_dataset()

        self.determine_target_model()
        self.determine_shadow_model()
        self.train_target_model()
        self.train_shadow_model()

    def determine_max_nodes(self):
        if self.args['exp'] in ['graph_recon', 'graph_recon_base', 'defense_perturb']:
            self.max_nodes = 20
        else:
            self.max_nodes = self.args['max_nodes']

    def load_data(self):
        self.dataset = self.data_store.load_raw_data(self.dataset_name)
        self.shadow_dataset = self.data_store.load_raw_data(self.args['shadow_dataset'])
        target_x = self.dataset.data.x.numpy()
        shadow_x = self.shadow_dataset.data.x.numpy()
        self.data_statistic(self.dataset)
        self.data_statistic(self.shadow_dataset)

    def data_statistic(self, dataset):
        self.logger.info("DatasetName: %s,"
                         "Number of graphs: %s, "
                         "Average Nodes: %s,"
                         "Average Edges: %s,"
                         "Node Features: %s, "
                         "Number of classes: %s," %
                         (dataset.name,
                         len(dataset.data.num_nodes),
                          np.mean(dataset.data.num_nodes),
                          dataset.data.num_edges/len(dataset.data.num_nodes)/2,
                          dataset.num_node_features,
                          dataset.num_classes,))

    def determine_target_model(self):
        if self.args['target_model'] == 'diff_pool':
            self.target_model = DiffPool(self.dataset.num_features, self.dataset.num_classes, self.max_nodes, self.args)
        elif self.args['target_model'] == 'mincut_pool':
            self.target_model = MinCutPool(self.dataset.num_features, self.dataset.num_classes, self.max_nodes, self.args)
        elif self.args['target_model'] == 'mean_pool':
            self.target_model = MeanPool(self.dataset.num_features, self.dataset.num_classes, self.args)
        else:
            raise Exception('unsupported target model')

    def determine_shadow_model(self):
        if self.args['shadow_model'] == 'diff_pool':
            self.shadow_model = DiffPool(self.dataset.num_features, self.dataset.num_classes, self.max_nodes, self.args)
        elif self.args['shadow_model'] == 'mincut_pool':
            self.shadow_model = MinCutPool(self.dataset.num_features, self.dataset.num_classes, self.max_nodes, self.args)
        elif self.args['shadow_model'] == 'mean_pool':
            self.shadow_model = MeanPool(self.dataset.num_features, self.dataset.num_classes, self.args)
        else:
            raise Exception('unsupported shadow model')

    def split_data(self):
        if self.args['is_split'] or not os.path.exists(self.data_store.split_file):
            self.logger.debug('splitting data')

            num_total_graphs = len(self.dataset)
            num_target_graphs = int(num_total_graphs * self.args['target_ratio'])
            num_shadow_graphs = int(num_total_graphs * self.args['shadow_ratio'])
            num_attack_train_graphs = int(num_total_graphs * self.args['attack_train_ratio'])

            self.target_indices = np.random.choice(np.arange(num_total_graphs), num_target_graphs, replace=False)
            remain_user_indices = np.setdiff1d(np.arange(num_total_graphs), self.target_indices)

            self.shadow_indices = np.random.choice(remain_user_indices, num_shadow_graphs, replace=False)
            remain_user_indices = np.setdiff1d(remain_user_indices, self.shadow_indices)

            self.attack_train_indices = np.random.choice(remain_user_indices, num_attack_train_graphs, replace=False)

            self.attack_test_indices = np.setdiff1d(remain_user_indices, self.attack_train_indices)

            self.data_store.save_split_data((self.target_indices, self.shadow_indices, self.attack_train_indices, self.attack_test_indices))
        else:
            self.target_indices, self.shadow_indices, self.attack_train_indices, self.attack_test_indices = self.data_store.load_split_data()

    def gen_attack_dataset(self):
        if self.args['dataset_name'] == self.args['shadow_dataset']:
            ### DHFR -> DHFR
            # diffpool: auc = 0.71; meanpool: auc = 0.93
            self.attack_train_dataset = self.dataset[list(self.attack_train_indices)]
            self.sub_train_neg_dataset = self.dataset[list(self.attack_test_indices)]

            self.attack_test_dataset = self.dataset[list(self.attack_test_indices)]
            self.sub_test_neg_dataset = self.dataset[list(self.attack_train_indices)]
        else:
            self.logger.info('target dataset and shadow dataset are different')
            attack_train_indices = np.random.choice(np.arange(len(self.shadow_dataset)), int(len(self.shadow_dataset) / 2))
            attack_test_indices = np.setdiff1d(np.arange(len(self.shadow_dataset)), attack_train_indices)

            self.attack_train_dataset = self.shadow_dataset[list(attack_train_indices)]
            self.sub_train_neg_dataset = self.shadow_dataset[list(attack_test_indices)]
            self.attack_test_dataset = self.dataset[list(self.attack_test_indices)]
            self.sub_test_neg_dataset = self.dataset[list(self.attack_train_indices)]

    def train_target_model(self):
        if self.args['is_train_target_model']:
            self.logger.info('training target model')

            target_train_dataset = self.dataset[list(self.target_indices)]
            # target_test_dataset = self.dataset[list(self.shadow_indices)]
            target_test_dataset = self.dataset[list(self.attack_test_indices)]
            target_train_loader = DenseDataLoader(target_train_dataset, batch_size=self.args['batch_size'])
            target_test_loader = DenseDataLoader(target_test_dataset, batch_size=self.args['batch_size'])

            self.target_model.train_model(target_train_loader, target_test_loader, self.args['num_epochs'])
            self.data_store.save_target_model(self.target_model)

    def train_shadow_model(self):
        if self.args['is_train_shadow_model']:
            self.logger.info('training shadow model')

            shadow_train_dataset = self.dataset[list(self.shadow_indices)]
            shadow_test_dataset = self.dataset[list(self.target_indices)]
            shadow_train_loader = DenseDataLoader(shadow_train_dataset, batch_size=self.args['batch_size'])
            shadow_test_loader = DenseDataLoader(shadow_test_dataset, batch_size=self.args['batch_size'])

            self.shadow_model.train_model(shadow_train_loader, shadow_test_loader, self.args['num_epochs'])
            self.data_store.save_shadow_model(self.shadow_model)
