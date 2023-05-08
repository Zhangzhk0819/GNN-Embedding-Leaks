import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

from exp.exp import Exp
from lib_property_infer.attack import Attack
from lib_subgraph_infer.attack_subgraph_infer import AttackSubgraphInfer
from lib_graph_recon.attack_graph_recon import AttackGraphRecon
from lib_classifier.mlp import MLP


class ExpDefensePerturb(Exp):
    def __init__(self, args):
        super(ExpDefensePerturb, self).__init__(args)
        self.logger = logging.getLogger('exp_defense_perturb')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args['attack'] == 'property_infer':
            self.property_infer_defense()
        elif args['attack'] == 'subgraph_infer_2':
            self.subgraph_infer_defense()
        elif args['attack'] == 'graph_recon':
            self.graph_recon_defense()

    def property_infer_defense(self):
        self.logger.info('defending property inference')

        paras = self.data_store.load_target_model(self.target_model)
        attack = Attack(self.target_model.model, self.target_model.model, self.args)

        attack.generate_train_embedding(self.attack_train_dataset, paras['embedding_dim'])
        attack.generate_test_embedding(self.attack_test_dataset, paras['embedding_dim'])
        attack.generate_labels(self.attack_train_dataset, self.attack_test_dataset, self.args['property_num_class'])
        attack.train_attack_model()

        self.property_infer_result = {}

        for noise_std in np.linspace(0.0, 10.0, 11):
            attack_acc = np.zeros(self.args['num_runs'])
            original_acc = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                # perturb test embedding
                target_embedding = attack.test_graph_embedding
                perturb_embedding = self._embeddings_perturb(target_embedding, noise_std)
                attack.test_graph_embedding = perturb_embedding

                # evaluate perturbed embedding
                # attack_acc[run] = attack.evaluate_attack_model()['radius']
                attack_acc[run] = attack.evaluate_attack_model()['density']

                # evaluate original task
                original_acc[run] = self._original_task_acc(perturb_embedding, self.attack_test_dataset.data.y[list(self.attack_test_indices)],
                                                        attack.train_graph_embedding, self.attack_train_dataset.data.y[list(self.attack_train_indices)])

            self.property_infer_result[noise_std] = [[np.mean(attack_acc), np.std(attack_acc)],
                                                     [np.mean(original_acc), np.std(original_acc)]]
            self.logger.info("noise_std: %f, attack acc: %s, original acc: %s" % (noise_std, np.mean(attack_acc), np.mean(original_acc)))

        self.data_store.save_property_infer_defense_data(self.property_infer_result)

    def subgraph_infer_defense(self):
        self.logger.info('defending subgraph inference')

        paras = self.data_store.load_target_model(self.target_model)
        attack = AttackSubgraphInferII(self.target_model.model, self.target_model.model, paras['embedding_dim'], self.dataset.num_classes, self.args)

        is_train_attack_model = True
        if is_train_attack_model:
            attack.determine_subsample_cls(self.args['train_sample_method'])
            attack.generate_train_data(self.attack_train_dataset, self.sub_train_neg_dataset)
            attack.determine_subsample_cls(self.args['test_sample_method'])
            attack.generate_test_data(self.attack_test_dataset, self.sub_test_neg_dataset)

            attack.determine_feat_gen_fn(self.args['feat_gen_method'][0])
            attack.generate_dataloader()
            attack.train_attack_model(self.dataset.num_features, self.args['feat_gen_method'][0])
        else:
            self.data_store.load_subgraph_infer_2_data(attack)
            attack.train_attack_model(self.dataset.num_features, self.args['feat_gen_method'][0], is_train=False)
            self.data_store.load_subgraph_infer_2_model(attack)

        self.subgraph_infer_result = {}

        for noise_std in np.linspace(0.0, 10.0, 11):
            attack_auc = np.zeros(self.args['num_runs'])
            original_acc = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                # perturb test embedding
                target_embedding = attack.test_graph_embedding
                perturb_embedding = self._embeddings_perturb(target_embedding, noise_std)
                attack.test_graph_embedding = perturb_embedding

                # evaluate perturbed embedding
                attack.generate_dataloader()
                _, attack_auc[run] = attack.evaluate_attack_model()

                # evaluate original task
                original_acc[run] = self._original_task_acc(perturb_embedding, self.attack_test_dataset.data.y[list(self.attack_test_indices)],
                                                            attack.train_graph_embedding, self.attack_train_dataset.data.y[list(self.attack_train_indices)])

            self.subgraph_infer_result[noise_std] = [[np.mean(attack_auc), np.std(attack_auc)],
                                                     [np.mean(original_acc), np.std(original_acc)]]
            self.logger.info("noise_std: %f, attack auc: %s, original acc: %s" % (noise_std, np.mean(attack_auc), np.mean(original_acc)))

        self.data_store.save_subgraph_infer_2_defense_data(self.subgraph_infer_result)

    def graph_recon_defense(self):
        self.logger.info('defending graph reconstruction')

        paras = self.data_store.load_target_model(self.target_model)
        attack = AttackGraphRecon(self.target_model.model, self.max_nodes, self.args)
        attack.init_graph_vae(self.dataset, paras['embedding_dim'], self.max_nodes)

        self.data_store.load_graph_vae_model(attack)
        self.graph_recon_result = {}

        for noise_std in np.linspace(0.0, 10.0, 11):
            metric_value = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                # perturb test embedding
                attack.gen_test_embedding(self.attack_test_dataset)
                target_embedding = attack.test_graph_embedding
                perturb_embedding = self._embeddings_perturb(target_embedding.detach().cpu().numpy(), noise_std)
                attack.test_graph_embedding = torch.from_numpy(perturb_embedding).to(self.device)
                attack.reconstruct_graph()

                # evaluate perturbed embedding
                # todo: can change statistic and metric here
                # attack.determine_stat('isomorphism_test')
                attack.determine_stat('degree_dist')
                attack.determine_metric('cosine_similarity')
                metric_value[run] = attack.evaluate_reconstruction(self.attack_test_dataset, attack.recon_adjs)

            self.graph_recon_result[noise_std] = [np.mean(metric_value), np.std(metric_value)]

        self.data_store.save_graph_recon_defense_data(self.graph_recon_result)

    def _original_task_acc(self, train_embedding, train_label, test_embedding, test_label):
        # notice that the train and test here is opposite to the attack model
        train_dset = self._gen_tensor_dataset(train_embedding, train_label)
        test_dset = self._gen_tensor_dataset(test_embedding, test_label)

        original_model = MLP(train_embedding.shape[1], self.dataset.num_classes)
        original_model.train_model(train_dset, num_epochs=100)
        acc = original_model.calculate_acc(test_dset, None)

        return acc

    def _embeddings_perturb(self, embeddings, noise_std):
        ret_embeddings = np.zeros_like(embeddings)

        for i, embedding in enumerate(embeddings):
            ret_embeddings[i] = embedding + np.random.laplace(loc=0.0, scale=noise_std, size=embedding.size)

        return ret_embeddings

    def _gen_tensor_dataset(self, feat, label):
        train_x = torch.tensor(np.int64(feat)).float()
        # train_y = torch.tensor(np.int64(label))
        return TensorDataset(train_x, label)

    def upload_data(self, operate_db, upload_data):
        pass
