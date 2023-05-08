import logging
from collections import defaultdict

import torch
import numpy as np

from exp.exp import Exp
from lib_graph_recon.attack_graph_recon import AttackGraphRecon


class ExpGraphRecon(Exp):
    def __init__(self, args):
        super(ExpGraphRecon, self).__init__(args)

        self.logger = logging.getLogger('exp_graph_recon')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.graph_recon_stat_run = []
        self.graph_recon_stat = defaultdict(dict)
        self.graph_recon_dist = defaultdict(dict)

        self.train_gae()

        if not self.args['is_ablation']:
            self.launch_attack()
            self.cal_stat()
        else:
            self.ablation_study()

    def train_gae(self):
        self.logger.info('launching attack')

        # load target model and its corresponding parameters
        if not self.args['is_train_target_model']:
            paras = self.data_store.load_target_model(self.target_model)
        else:
            paras = self.target_model.paras

        self.attack = AttackGraphRecon(self.target_model.model, self.max_nodes, self.args)
        self.attack.init_graph_vae(self.dataset, paras['embedding_dim'], self.max_nodes)

        # limit the number of graphs, since the computational cost of graphvae is high
        if len(self.attack_train_dataset) > 500:
            select = np.random.choice(np.arange(len(self.attack_train_dataset)), 500)
            self.attack_train_dataset = self.attack_train_dataset[select.tolist()]

        if len(self.attack_test_dataset) > 500:
            select = np.random.choice(np.arange(len(self.attack_test_dataset)), 500)
            self.attack_test_dataset = self.attack_test_dataset[select.tolist()]

        # step1: train gae model
        if self.args['is_train_gae']:
            self.attack.train_gae(self.attack_train_dataset, num_epoch=self.args['gae_num_epochs'])
            self.data_store.save_graph_vae_model(self.attack)
        else:
            self.data_store.load_graph_vae_model(self.attack)

        # step2: fine-tune gae model
        if self.args['is_use_fine_tune']:
            if self.args['is_fine_tune_gae']:
                self.attack.gen_fine_tune_dataset(self.attack_train_dataset, paras['embedding_dim'])
                self.attack.fine_tune_gae(self.args['fine_tune_num_epochs'])
                self.data_store.save_graph_vae_finetune_model(self.attack)
            else:
                self.data_store.load_graph_vae_finetune_model(self.attack)

    def launch_attack(self):
        for run in range(self.args['num_runs']):
            self.logger.info('%s run' % (run,))

            # step3: generate reconstruction data
            if self.args['is_gen_recon_data']:
                self.attack.gen_test_embedding(self.attack_test_dataset)
                self.attack.reconstruct_graph()
                self.data_store.save_graph_recon_data(self.attack)
            else:
                self.data_store.load_graph_recon_data(self.attack)

            # step4: evaluate gae model
            graph_recon_stat_data = defaultdict(dict)
            for graph_recon_stat in self.args['graph_recon_stat']:
                self.attack.determine_stat(graph_recon_stat)
                for graph_recon_metric in self.args['graph_recon_metric']:
                    self.attack.determine_metric(graph_recon_metric)

                    metric_value = self.attack.evaluate_reconstruction(self.attack_test_dataset, self.attack.recon_adjs)
                    graph_recon_stat_data[graph_recon_stat][graph_recon_metric] = metric_value

            self.graph_recon_stat_run.append(graph_recon_stat_data)

    def ablation_study(self):
        for num_epoch in range(0, self.args['gae_num_epochs'], self.args['epoch_step']):
            self.num_epoch = num_epoch
            self.data_store.load_graph_vae_model_epoch(self.attack, num_epoch)

            for run in range(self.args['num_runs']):
                self.logger.info('%s run' % (run,))

                # step3: generate reconstruction data
                self.attack.gen_test_embedding(self.attack_test_dataset)
                self.attack.reconstruct_graph()

                # step4: evaluate gae model
                graph_recon_stat_data = defaultdict(dict)
                for graph_recon_stat in self.args['graph_recon_stat']:
                    self.attack.determine_stat(graph_recon_stat)
                    for graph_recon_metric in self.args['graph_recon_metric']:
                        self.attack.determine_metric(graph_recon_metric)

                        metric_value = self.attack.evaluate_reconstruction(self.attack_test_dataset, self.attack.recon_adjs)
                        graph_recon_stat_data[graph_recon_stat][graph_recon_metric] = metric_value

                self.graph_recon_stat_run.append(graph_recon_stat_data)

            self.cal_stat()
            self.upload_stat()

    def cal_stat(self):
        self.logger.info('calculating statistics')

        for graph_recon_stat in self.args['graph_recon_stat']:
            for graph_recon_metric in self.args['graph_recon_metric']:

                run_data = np.zeros(self.args['num_runs'])
                for run in range(self.args['num_runs']):
                    run_data[run] = self.graph_recon_stat_run[run][graph_recon_stat][graph_recon_metric]

                metric_value = [np.mean(run_data), np.std(run_data)]
                self.graph_recon_stat[graph_recon_stat][graph_recon_metric] = metric_value

                self.logger.info('graph_recon_stat: %s, graph_recon_metric: %s, %s' %
                                 (graph_recon_stat, graph_recon_metric, metric_value))
