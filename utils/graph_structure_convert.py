import logging
import os.path as osp

import networkx as nx
from lib_dataset.tu_dataset import TUDataset
import torch_geometric.transforms as T

import config


class GraphStructureConvert:
    def __init__(self):
        self.logger = logging.getLogger('graph_structure_convert')
        
    def pyg2nx(self, pyg_graph):
        pass
    
    def nx2pyg(self, nx_graph):
        pass


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= 1000


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'DD'
    path = osp.join(config.ORIGINAL_DATASET_PATH, dataset_name)
    dataset = TUDataset(path, name=dataset_name, use_node_attr=True, use_edge_attr=False, transform=T.ToDense(1000), pre_filter=MyFilter())
    convert = GraphStructureConvert()
    
    nx_data = convert.pyg2nx(dataset[0])
    pyg_data = convert.nx2pyg(nx_data)
