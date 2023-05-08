import torch
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


class Classifier:
    def __init__(self):
        self.name = None
        self.model = None
        self.cuda_avail = None

    def save_model(self, save_name):
        if self.name in ['dnn', 'mlp']:
            torch.save(self.model.state_dict(), save_name)
        elif self.name in ['lr', 'dt']:
            joblib.dump(self.model, save_name, compress=9)
        elif self.name == 'multi_class_classifier':
            torch.save(self.model, save_name)
        else:
            raise Exception('invalid classifier')
    
    def load_model(self, save_name):
        if self.name in ['dnn', 'mlp']:
            self.model.load_state_dict(torch.load(save_name))
        elif self.name in ['lr', 'dt']:
            self.model = joblib.load(save_name)
        elif self.name == 'multi_class_classifier':
            self.model = torch.load(save_name)
        else:
            raise Exception('invalid classifier')
    
    def predict_proba(self, test_x):
        pass
    
    def calculate_multi_class_acc(self, test_x):
        ret_acc = {}
        posteriors, labels = self.predict_proba(test_x)
        
        for attr, post in posteriors.items():
            post, true_y = post.cpu().detach().numpy(), labels[attr].cpu().detach().numpy()
            pred_y = np.argmax(post.data, 1)
            ret_acc[attr] = accuracy_score(true_y, pred_y)
            
        return ret_acc

    def calculate_acc(self, test_x, test_y):
        if self.name in ['dnn', 'mlp']:
            posteriors, test_y = self.predict_proba(test_x)
            posteriors, test_y = posteriors.cpu().detach().numpy(), test_y.cpu().detach().numpy()
            pred_y = np.argmax(posteriors.data, 1)
        else:
            posteriors = self.predict_proba(test_x)
            pred_y = np.argmax(posteriors.data, 1)
    
        return accuracy_score(test_y, pred_y)
    
    def calculate_auc(self, test_x, test_y):
        if self.name in ['dnn', 'mlp']:
            posteriors, test_y = self.predict_proba(test_x)
            posteriors, test_y = posteriors.cpu().detach().numpy(), test_y.cpu().detach().numpy()
        else:
            posteriors = self.predict_proba(test_x)
    
        return roc_auc_score(test_y, posteriors[:, 1])
