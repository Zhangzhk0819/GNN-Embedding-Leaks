from torch.utils.data.dataset import Dataset


class FineTuneDataset(Dataset):
    def __init__(self, dataset, embedding):
        self.dataset = dataset
        self.embedding = embedding
    
    def __getitem__(self, index):
        return self.dataset[index].adj, self.dataset[index].mask, self.embedding[index]
    
    def __len__(self):
        return len(self.dataset)
