import logging
import os

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from lib_classifier.nn_structure.mlp_net import MLPNet
from lib_classifier.classifier import Classifier


class MLP(Classifier):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.logger = logging.getLogger('mlp')
        self.name = 'mlp'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MLPNet(input_size, num_classes).to(self.device)

    def train_model(self, train_dset, num_epochs=100):
        torch.set_num_threads(1)
        self.model.train()

        # define training components
        train_loader = DataLoader(dataset=train_dset, batch_size=32, shuffle=True)
        # optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        parameters = [para for para in self.model.parameters()]
        optimizer = SGD(parameters, lr=0.01, weight_decay=0.001)
        # loss_fn = nn.CrossEntropyLoss()

        # training process
        for epoch in range(num_epochs):
            self.logger.debug("epoch %s" % (epoch,))

            for i, (feats, labels) in enumerate(train_loader):
                feats, labels = feats.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(feats)
                loss = self.model.loss(outputs, labels)
                # loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict_proba(self, test_dset):
        self.model.eval()
        start = True

        with torch.no_grad():
            test_loader = DataLoader(dataset=test_dset, batch_size=32, shuffle=True)

            for feats, labels in test_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)

                posterior = self.model(feats)
                if start:
                    ret_posteriors = posterior
                    ret_labels = labels
                    start = False
                else:
                    ret_posteriors = torch.cat((ret_posteriors, posterior), dim=0)
                    ret_labels = torch.cat((ret_labels, labels), dim=0)

            return ret_posteriors, ret_labels


if __name__ == '__main__':
    os.chdir("../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
