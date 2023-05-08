import os.path as osp

from lib_dataset.tu_dataset import TUDataset
import torch_geometric.transforms as T

import config


class LoadData:
    def __init__(self):
        pass
    
    def load_dataset(self, dataset_name, max_nodes):
        path = osp.join(config.ORIGINAL_DATASET_PATH, dataset_name)
        return TUDataset(path, name=dataset_name, use_node_attr=True, use_edge_attr=False,
                         transform=T.ToDense(max_nodes), pre_filter=MyFilter(max_nodes))


class MyFilter(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
    
    def __call__(self, data):
        return data.num_nodes <= self.max_nodes
