import logging

import numpy as np

from exp.exp import Exp
from lib_subgraph_infer.attack_subgraph_infer import AttackSubgraphInfer


class ExpSubgraphInfer(Exp):
    def __init__(self, args):
        super(ExpSubgraphInfer, self).__init__(args)
        self.logger = logging.getLogger('exp_subgraph_infer_2')

        self.acc_run = []
        self.auc_run = []
        self.acc = {}
        self.auc = {}

        self.launch_attack()
        self.cal_stat()

    def launch_attack(self):
        self.logger.info('launching attack')

        # load target model and its corresponding parameters
        if not self.args['is_train_target_model']:
            paras = self.data_store.load_target_model(self.target_model)
        else:
            paras = self.target_model.paras

        # load shadow model and its corresponding parameters
        if self.args['is_use_shadow_model']:
            if not self.args['is_train_shadow_model']:
                paras = self.data_store.load_shadow_model(self.shadow_model)
            else:
                paras = self.shadow_model.paras

        if self.args['is_use_shadow_model']:
            attack = AttackSubgraphInfer(self.target_model.model, self.shadow_model.model, paras['embedding_dim'], self.dataset.num_classes, self.args)
        else:
            attack = AttackSubgraphInfer(self.target_model.model, self.target_model.model, paras['embedding_dim'], self.dataset.num_classes, self.args)

        for run in range(self.args['num_runs']):
            self.logger.info('%s run' % (run,))
            # generate attack training data
            if self.args['is_gen_attack_data']:
                # attack_train_dataset = self.dataset[list(self.attack_train_indices)]
                # attack_test_dataset = self.dataset[list(self.attack_test_indices)]

                attack.determine_subsample_cls(self.args['train_sample_method'])
                # attack.generate_train_data(self.attack_train_dataset, self.attack_test_dataset)
                attack.generate_train_data(self.attack_train_dataset, self.sub_train_neg_dataset)

                attack.determine_subsample_cls(self.args['test_sample_method'])
                # attack.generate_test_data(self.attack_test_dataset, self.attack_train_dataset)
                attack.generate_test_data(self.attack_test_dataset, self.sub_test_neg_dataset)

                self.data_store.save_subgraph_infer_2_data(attack)
            else:
                self.data_store.load_subgraph_infer_2_data(attack)

            acc, auc = {}, {}
            # train and test attack model
            for feat_gen_method in self.args['feat_gen_method']:
                attack.determine_feat_gen_fn(feat_gen_method)
                attack.generate_dataloader()

                attack.train_attack_model(self.dataset.num_features, feat_gen_method)
                self.data_store.save_subgraph_infer_2_model(attack)
                acc[feat_gen_method], auc[feat_gen_method] = attack.evaluate_attack_model()

            self.acc_run.append(acc)
            self.auc_run.append(auc)

    def cal_stat(self):
        self.logger.info('calculating statistics')

        for feat_gen_method in self.args['feat_gen_method']:
            acc_run_data = np.zeros(self.args['num_runs'])
            auc_run_data = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                acc_run_data[run] = self.acc_run[run][feat_gen_method]
                auc_run_data[run] = self.auc_run[run][feat_gen_method]

            self.acc[feat_gen_method] = [np.mean(acc_run_data), np.std(acc_run_data)]
            self.auc[feat_gen_method] = [np.mean(auc_run_data), np.std(auc_run_data)]

            self.logger.info('config: %s, attack acc: %s, attack auc %s' % (
                feat_gen_method, self.acc[feat_gen_method], self.auc[feat_gen_method]))
