from lib_dataset.tu_dataset import TUDataset
import torch_geometric.transforms as T
import torch
import numpy as np

import config


class DataStore:
    def __init__(self, args, max_nodes):
        self.args = args

        self.dataset_name = args['dataset_name']
        self.shadow_dataset_name = args['shadow_dataset']
        self.target_model_name = args['target_model']
        self.shadow_model_name = args['shadow_model']
        self.max_nodes = max_nodes

        self.determine_data_name()
        self.determine_data_path()
        self.generate_folder()

    def determine_data_name(self):
        self.data_name = {}

        self.data_name['target_model_name'] = '_'.join(('model', self.dataset_name, self.target_model_name))
        self.data_name['target_para_name'] = '_'.join(('para', self.dataset_name, self.target_model_name))
        self.data_name['shadow_model_name'] = '_'.join(('model', self.dataset_name, self.shadow_model_name))
        self.data_name['shadow_para_name'] = '_'.join(('para', self.dataset_name, self.shadow_model_name))

        self.data_name['property_infer_name'] = '_'.join((self.dataset_name,
                                                          self.shadow_dataset_name,
                                                          self.target_model_name, self.shadow_model_name,
                                                          str(self.args['property_num_class'])))
        self.data_name['subgraph_infer_name'] = '_'.join((self.dataset_name,
                                                          self.shadow_dataset_name,
                                                          str(self.args['is_use_shadow_model']),
                                                          self.target_model_name, self.shadow_model_name,
                                                          str(self.args['train_sample_method']), str(self.args['test_sample_method']),
                                                          str(self.args['sample_node_ratio'])))
        self.data_name['graph_recon_name'] = '_'.join((self.dataset_name, self.args['encoder_method']))

        if not self.args['is_use_feat']:
            for data, name in self.data_name.items():
                self.data_name[data] = name + '_non_feat'

    def determine_data_path(self):
        self.split_file = config.SPLIT_PATH + str(self.max_nodes) + '/' + self.dataset_name

        self.target_model_file = config.TARGET_MODEL_PATH + str(self.max_nodes) + '/' + self.data_name['target_model_name']
        self.target_model_para_file = config.TARGET_MODEL_PATH + str(self.max_nodes) + '/' + self.data_name['target_para_name']
        self.shadow_model_file = config.SHADOW_MODEL_PATH + str(self.max_nodes) + '/' + self.data_name['shadow_model_name']
        self.shadow_model_para_file = config.SHADOW_MODEL_PATH + str(self.max_nodes) + '/' + self.data_name['shadow_para_name']

        self.property_infer_data_file = config.ATTACK_DATA_PATH + 'property_infer/' + self.data_name['property_infer_name']
        self.property_infer_model_file = config.ATTACK_MODEL_PATH + 'property_infer/' + self.data_name['property_infer_name']
        self.property_infer_defense_file = config.DEFENSE_DATA_PATH + 'property_infer/' + self.data_name['property_infer_name']

        self.subgraph_infer_1_data_file = config.ATTACK_DATA_PATH + 'subgraph_infer_1/' + self.data_name['subgraph_infer_name']
        self.subgraph_infer_1_model_file = config.ATTACK_MODEL_PATH + 'subgraph_infer_1/' + self.data_name['subgraph_infer_name']
        self.subgraph_infer_1_defense_file = config.DEFENSE_DATA_PATH + 'subgraph_infer_1/' + self.data_name['subgraph_infer_name']

        self.subgraph_infer_2_data_file = config.ATTACK_DATA_PATH + 'subgraph_infer_2/' + self.data_name['subgraph_infer_name']
        self.subgraph_infer_2_model_file = config.ATTACK_MODEL_PATH + 'subgraph_infer_2/' + self.data_name['subgraph_infer_name']
        self.subgraph_infer_2_defense_file = config.DEFENSE_DATA_PATH + 'subgraph_infer_2/' + self.data_name['subgraph_infer_name']

        self.graph_vae_model_file = config.GAE_MODEL_PATH + self.data_name['graph_recon_name']
        self.graph_vae_finetune_model_file = config.GAE_MODEL_PATH + 'fine_tune/' + self.data_name['graph_recon_name']
        self.graph_recon_data_file = config.ATTACK_DATA_PATH + 'graph_reconstruct/' + self.data_name['graph_recon_name']
        self.graph_recon_defense_file = config.DEFENSE_DATA_PATH + 'graph_recon/' + self.data_name['graph_recon_name']

    def generate_folder(self):
        pass

    def load_raw_data(self, dataset_name):
        if self.args['dataset_name'] in ['DD', 'PROTEINS']:
            pre_transform = T.Compose([
                T.OneHotDegree(100, cat=False)  # use only node degree as node feature.
            ])
        else:
            pre_transform = T.Compose([
                T.OneHotDegree(10, cat=False)  # use only node degree as node feature.
            ])

        if self.args['is_use_feat']:
            dataset = TUDataset(config.RAW_DATA_PATH + str(self.max_nodes) + '/', name=dataset_name, use_node_attr=True,
                         use_edge_attr=False, transform=T.ToDense(self.max_nodes), pre_filter=MyFilter(self.max_nodes))
        else:
            dataset = TUDataset(config.RAW_DATA_PATH + '/'.join((str(self.max_nodes), 'non_feat')) + '/', name=dataset_name,
                         use_node_attr=True, use_edge_attr=False, transform=T.ToDense(self.max_nodes),
                         pre_filter=MyFilter(self.max_nodes), pre_transform=pre_transform)

        if dataset_name in ['OVCAR-8H', 'PC-3', 'MOLT-4H']:
            select_indices = np.random.choice(np.arange(len(dataset)), int(0.1 * len(dataset)))
            dataset = dataset[list(select_indices)]

        if dataset_name == 'PC-3':
            dataset.data.x = dataset.data.x[:, :37]

        if dataset_name == 'OVCAR-8H':
            dataset.data.x = dataset.data.x[:, :65]

        return dataset

    def save_split_data(self, indices):
        torch.save(indices, self.split_file)

    def load_split_data(self):
        return torch.load(self.split_file)

    ################################## target / shadow model ##################################
    def save_target_model(self, target_model):
        target_model.save_model(self.target_model_file)
        target_model.save_paras(self.target_model_para_file)

    def load_target_model(self, target_model):
        target_model.load_model(self.target_model_file)
        paras = target_model.load_paras(self.target_model_para_file)
        return paras

    def save_shadow_model(self, shadow_model):
        shadow_model.save_model(self.shadow_model_file)
        shadow_model.save_paras(self.shadow_model_para_file)

    def load_shadow_model(self, shadow_model):
        shadow_model.load_model(self.shadow_model_file)
        paras = shadow_model.load_paras(self.shadow_model_para_file)
        return paras

    ############################### property inference ###############################
    def save_property_infer_data(self, attack):
        attack.save_data(self.property_infer_data_file)

    def load_property_infer_data(self, attack):
        attack.load_data(self.property_infer_data_file)

    def save_property_infer_model(self, attack):
        attack.save_attack_model(self.property_infer_model_file)

    def load_property_infer_model(self, attack):
        attack.load_attack_model(self.property_infer_model_file)

    def save_property_infer_defense_data(self, defense_data):
        torch.save(defense_data, self.property_infer_defense_file)

    def load_property_infer_defense_data(self):
        return torch.load(self.property_infer_defense_file)

    ################################ subgraph inference ################################
    def save_subgraph_infer_1_data(self, attack):
        attack.save_data(self.subgraph_infer_1_data_file)

    def load_subgraph_infer_1_data(self, attack):
        attack.load_data(self.subgraph_infer_1_data_file)

    def save_subgraph_infer_1_model(self, attack):
        attack.save_attack_model(self.subgraph_infer_1_model_file)

    def load_subgraph_infer_1_model(self, attack):
        attack.load_attack_model(self.subgraph_infer_1_model_file)

    ################################ subgraph inference ################################
    def save_subgraph_infer_2_data(self, attack):
        attack.save_data(self.subgraph_infer_2_data_file)

    def load_subgraph_infer_2_data(self, attack):
        attack.load_data(self.subgraph_infer_2_data_file)

    def save_subgraph_infer_2_model(self, attack):
        attack.save_attack_model(self.subgraph_infer_2_model_file)

    def load_subgraph_infer_2_model(self, attack):
        attack.load_attack_model(self.subgraph_infer_2_model_file)

    def save_subgraph_infer_2_defense_data(self, defense_data):
        torch.save(defense_data, self.subgraph_infer_2_defense_file)

    def load_subgraph_infer_2_defense_data(self):
        return torch.load(self.subgraph_infer_2_defense_file)

    ################################# graph reconstruction ##############################
    def save_graph_vae_model(self, attack):
        attack.save_model(self.graph_vae_model_file)

    def load_graph_vae_model(self, attack):
        attack.load_model(self.graph_vae_model_file)

    def save_graph_vae_model_epoch(self, graph_vae, epoch):
        graph_vae.save_model(self.graph_vae_model_file + '_' + str(epoch))

    def load_graph_vae_model_epoch(self, attack, epoch):
        attack.load_model(self.graph_vae_model_file + '_' + str(epoch))

    def save_graph_vae_finetune_model(self, attack):
        attack.save_model(self.graph_vae_finetune_model_file)

    def load_graph_vae_finetune_model(self, attack):
        attack.load_model(self.graph_vae_finetune_model_file)

    def save_graph_recon_data(self, attack):
        attack.save_data(self.graph_recon_data_file)

    def load_graph_recon_data(self, attack):
        attack.load_data(self.graph_recon_data_file)

    def save_graph_recon_defense_data(self, defense_data):
        torch.save(defense_data, self.graph_recon_defense_file)

    def load_graph_recon_defense_data(self):
        return torch.load(self.graph_recon_defense_file)


class MyFilter(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes
