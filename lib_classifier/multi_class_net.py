import torch
from torch import nn
import torch.nn.functional as F


class MultiClassNet(nn.Module):
    def __init__(self, input_size, classes_dict):
        super(MultiClassNet, self).__init__()
        
        self.classes_dict = classes_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.share_layers = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.Linear(250, 100),
        )
        
        self.output_layers = {}
        for attr, num_class in classes_dict.items():
            self.output_layers[attr] = nn.Linear(100, num_class)
            self.output_layers[attr].to(self.device)

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        
        x = self.share_layers(x)
        
        output = {}
        for attr, _ in self.classes_dict.items():
            logit = self.output_layers[attr](x)
            output[attr] = F.softmax(logit, dim=1)
        
        return output
    
    def loss(self, output, labels):
        loss = 0.0
        
        for attr, _ in self.classes_dict.items():
            loss += F.cross_entropy(output[attr], labels[attr])
            
        return loss
