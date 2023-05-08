import logging
from collections import defaultdict

import torch

from exp.exp import Exp
from lib_graph_recon.attack_graph_recon_base import AttackGraphReconBase


class ExpGraphReconBase(Exp):
    def __init__(self, args):
        super(ExpGraphReconBase, self).__init__(args)
        
        self.logger = logging.getLogger('exp_graph_recon_base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.graph_recon_stat = defaultdict(dict)
        
        self.launch_attack()
    
    def launch_attack(self):
        self.logger.info('launching attack')
        
        attack = AttackGraphReconBase(self.target_model.model, self.max_nodes, self.args)
        attack_train_dataset = self.dataset[list(self.attack_train_indices)]
        attack_test_dataset = self.dataset[list(self.attack_test_indices)]
        
        # generate graph
        attack.gen_graph(20, 2, 0.1)
        attack.gen_recon_adjs(len(attack_test_dataset))

        # evaluate gae model
        for graph_recon_stat in self.args['graph_recon_stat']:
            attack.determine_stat(graph_recon_stat)
            for graph_recon_metric in self.args['graph_recon_metric']:
                attack.determine_metric(graph_recon_metric)
                metric_value = attack.evaluate_reconstruction(attack_test_dataset, attack.recon_adjs)
                self.graph_recon_stat[graph_recon_stat][graph_recon_metric] = metric_value
        
                self.logger.info('graph_recon_stat: %s, graph_recon_metric: %s, %s' %
                                 (graph_recon_stat, graph_recon_metric, metric_value))
