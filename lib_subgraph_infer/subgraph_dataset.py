import torch
from torch.utils.data import Dataset


class SubgraphDataset(Dataset):
    def __init__(self, graph_embedding, positive_subgraph, negative_subgraph):
        self.graph_embedding = graph_embedding
        self.positive_subgraph = positive_subgraph
        self.negative_subgraph = negative_subgraph
        
        self._preprocess()
    
    def __getitem__(self, index):
        return self.x[index], self.adj[index], self.mask[index], self.graph_embedding[index], self.labels[index]
    
    def __len__(self):
        return self.graph_embedding.shape[0]
    
    def _preprocess(self):
        x, adj, mask, labels = [], [], [], []
        
        for i in range(self.graph_embedding.shape[0]):
            x.append(self.positive_subgraph[i]['x'])
            adj.append(self.positive_subgraph[i]['adj'])
            mask.append(self.positive_subgraph[i]['mask'])

        for i in range(self.graph_embedding.shape[0]):
            x.append(self.negative_subgraph[i]['x'])
            adj.append(self.negative_subgraph[i]['adj'])
            mask.append(self.negative_subgraph[i]['mask'])
            
        self.x = torch.stack(x)
        self.adj = torch.stack(adj)
        self.mask = torch.stack(mask)
        self.labels = torch.cat((torch.ones(self.graph_embedding.shape[0]), torch.zeros(self.graph_embedding.shape[0]))).long()
        self.graph_embedding = torch.cat((torch.from_numpy(self.graph_embedding), torch.from_numpy(self.graph_embedding)))
