import networkx as nx

from lib_graph_recon.attack import Attack


class AttackGraphReconBase(Attack):
    def __init__(self, target_model, max_nodes, args):
        super(AttackGraphReconBase, self).__init__(target_model, max_nodes, args)
        
    def gen_graph(self, n, m, p):
        self.logger.info('generating graph')

        if self.args['graph_gen_method'] == 'BA':
            self.graph = self._gen_ba_graph(n, m)
        elif self.args['graph_gen_method'] == 'ER':
            self.graph = self._gen_er_graph(n, p)
        else:
            raise Exception('unsupported graph generation method')

    def gen_recon_adjs(self, num_adj):
        adj = nx.linalg.adj_matrix(self.graph).toarray()
        self.recon_adjs = [adj for _ in range(num_adj)]
        
    def _gen_ba_graph(self, n, m):
        return nx.generators.random_graphs.barabasi_albert_graph(n, m)
    
    def _gen_er_graph(self, n, p):
        return nx.generators.random_graphs.erdos_renyi_graph(n, p)
