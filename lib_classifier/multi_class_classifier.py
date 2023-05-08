import logging
import os

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from lib_classifier.classifier import Classifier
from lib_classifier.multi_class_net import MultiClassNet


class MultiClassClassifier(Classifier):
    def __init__(self, input_size, num_classes, index_attr_mapping):
        super(MultiClassClassifier, self).__init__()
        
        self.logger = logging.getLogger('dnn')
        self.name = 'multi_class_classifier'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.num_classes = num_classes
        self.attr_index_mapping = index_attr_mapping

        self.model = MultiClassNet(self.input_size, self.num_classes)

        self.model.to(self.device)
        for attr, num_class in self.model.classes_dict.items():
            self.model.output_layers[attr].to(self.device)

    def train_model(self, train_dset, num_epochs=10):
        torch.set_num_threads(1)
        self.model.train()

        # define training components
        train_loader = DataLoader(dataset=train_dset, batch_size=32, shuffle=True)
        # optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        parameters = [para for para in self.model.parameters()] + [para for key, value in self.model.output_layers.items() for para in value.parameters()]
        optimizer = SGD(parameters, lr=0.01, weight_decay=0.001)
        # loss_fn = nn.CrossEntropyLoss()

        # training process
        for epoch in range(num_epochs):
            self.logger.debug("epoch %s" % (epoch,))

            for i, (feats, labels) in enumerate(train_loader):
                feats, labels = feats.to(self.device), labels.to(self.device)
                if isinstance(self.num_classes, dict):
                    labels = self._index_attr_map(labels)

                optimizer.zero_grad()
                outputs = self.model(feats)
                loss = self.model.loss(outputs, labels)
                # loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict_proba(self, test_dset):
        self.model.eval()
        start = True

        self.model.to(self.device)
        for attr, num_class in self.model.classes_dict.items():
            self.model.output_layers[attr].to(self.device)

        with torch.no_grad():
            test_loader = DataLoader(dataset=test_dset, batch_size=32, shuffle=True)

            ret_posteriors = {}
            ret_labels = {}

            for feats, labels in test_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)

                labels = self._index_attr_map(labels)

                posterior = self.model(feats)

                if start:
                    for attr, post in posterior.items():
                        ret_posteriors[attr] = post
                        ret_labels[attr] = labels[attr]
                    start = False
                else:
                    for attr, post in posterior.items():
                        ret_posteriors[attr] = torch.cat((ret_posteriors[attr], post), dim=0)
                        ret_labels[attr] = torch.cat((ret_labels[attr], labels[attr]), dim=0)

            return ret_posteriors, ret_labels

    def _index_attr_map(self, labels):
        ret_labels = {}

        for idx in range(labels.shape[1]):
            ret_labels[self.attr_index_mapping[idx]] = labels[:, idx]

        return ret_labels


if __name__ == '__main__':
    os.chdir("../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)