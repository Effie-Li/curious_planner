import numpy as np
import networkx as nx

class NetworkWorld:
    
    '''
    a world for traversal on a network structure
    '''
    
    def __init__(self, 
                 nodes, 
                 edges,
                 action_dim):
        self.nodes = nodes # an array of nodes
        self.edges = edges # a dict of (node - list of nodes) paring
        self.current = None # for traversal
        self.action_dim = action_dim
        self.build_graph()
    
    def build_graph(self):
        self.graph = nx.DiGraph()
        for i in self.nodes:
            self.graph.add_node(i)
            for j in self.edges[i]:
                self.graph.add_edge(i, j, weight=1)
                
    def get_neighbors(self, node, lookup='neighbors'):
        '''
        returs the neighbors for the given node
        :param lookup: if neighbors, get all neighbors
                       if predecessors, get previous nodes
                       if successors, get next nodes
        '''
        if lookup=='neighbors':
            return list(self.graph.predecessors(node)) + list(self.graph.successors(node))
        if lookup=='predecessors':
            return list(self.graph.predecessors(node))
        if lookup=='successors':
            return list(self.graph.successors(node))
                
    def reset(self, start):
        self.current = start if start is not None else np.random.choice(self.nodes)
        return self.current

    def step(self, action):
        next_node = self.edges[self.current][action]
        self.current = next_node
        return self.current
    
    def shortest_path(self, s1, s2):
        shortest_path = nx.shortest_path(self.graph, s1, s2)
        actions = [self.edges[shortest_path[i-1]].index(shortest_path[i]) 
                   for i in np.arange(len(shortest_path))[1:]]
        n = len(actions)
        return {'n_step':n, 'path':shortest_path, 'actions':actions}
