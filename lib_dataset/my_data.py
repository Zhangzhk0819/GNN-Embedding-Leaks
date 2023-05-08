from torch_geometric.data import Data


class MyData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None):
        super(MyData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                 pos=pos, normal=normal, face=face)
        
    @property
    def density(self):
        return 0