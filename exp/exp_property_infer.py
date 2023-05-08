import logging

import numpy as np

from exp.exp import Exp
from lib_property_infer.attack import Attack


class ExpPropertyInfer(Exp):
    def __init__(self, args):
        super(ExpPropertyInfer, self).__init__(args)

        self.logger = logging.getLogger('exp_property_infer')

        self.properties = args['properties']
        self.property_num_class = args['property_num_class']

        self.acc_run = []
        self.acc = {}

        self.baseline_acc_run = []
        self.baseline_acc = {}

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
            attack = Attack(self.target_model.model, self.shadow_model.model, self.args)
        else:
            attack = Attack(self.target_model.model, self.target_model.model, self.args)

        # attack_train_dataset = self.dataset[list(self.attack_train_indices)]
        # attack_test_dataset = self.dataset[list(self.attack_test_indices)]

        for run in range(self.args['num_runs']):
            self.logger.info('%s run' % (run,))
            # generate attack training data
            if self.args['is_gen_embedding']:
                attack.generate_train_embedding(self.attack_train_dataset, paras['embedding_dim'])
                attack.generate_test_embedding(self.attack_test_dataset, paras['embedding_dim'])

                self.data_store.save_property_infer_data(attack)
            else:
                self.data_store.load_property_infer_data(attack)

            # train and test attack model
            attack.generate_labels(self.attack_train_dataset, self.attack_test_dataset, self.args['property_num_class'])
            attack.train_attack_model()
            self.data_store.save_property_infer_model(attack)

            self.acc_run.append(attack.evaluate_attack_model())
            self.baseline_acc_run.append(attack.baseline_acc)

    def cal_stat(self):
        self.logger.info('calculating statistics')

        for property in self.properties:
            run_data = np.zeros(self.args['num_runs'])
            baseline_run_data = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                run_data[run] = self.acc_run[run][property]
                baseline_run_data[run] = self.baseline_acc_run[run][property]

            self.acc[property] = [np.mean(run_data), np.std(run_data)]
            self.baseline_acc[property] = [np.mean(baseline_run_data), np.std(baseline_run_data)]

        self.logger.info('attack accuracy: %s' % (self.acc,))
        self.logger.info('baseline attack accuracy: %s' % (self.baseline_acc,))
