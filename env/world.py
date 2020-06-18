import numpy as np
import networkx as nx

class NetworkWorld:
    
    '''
    a world for traversal on a network structure
    '''
    
    def __init__(self, 
                 nodes, 
                 edges,
                 action_dim,
                 permute_node_labels=False):
        self.nodes = nodes # an array of nodes
        self.edges = edges # a dict of (node - list of nodes) paring
        self.current = None # for traversal
        self.action_dim = action_dim
        self.node_mapping = None
        if permute_node_labels:
            self.permute_node_labels()
        self.build_graph()

    def permute_node_labels(self):
        new_labels = np.copy(self.nodes)
        np.random.shuffle(new_labels)
        old_new_lookup = {}
        for i, old in enumerate(self.nodes):
            old_new_lookup[old] = new_labels[i]
        self.node_mapping = old_new_lookup
        # modify self.edges using the new nodes
        new_edges = {}
        for i, old in enumerate(self.nodes):
            new = old_new_lookup[old]
            old_successors = self.edges[old]
            new_successors = [old_new_lookup[o] for o in old_successors]
            new_edges[new] = new_successors
        self.edges = new_edges
        self.nodes = new_labels
    
    def build_graph(self):
        self.graph = nx.DiGraph()
        for i in self.nodes:
            self.graph.add_node(i)
            for j in self.edges[i]:
                self.graph.add_edge(i, j, weight=1)
                
    def get_neighbors(self, node, lookup='a'):
        '''
        returs the neighbors for the given node
        :param lookup: if a, get all neighbors
                       if p, get predecessors nodes
                       if s, get successors nodes
        '''
        if lookup=='a': # all neighbors
            return list(self.graph.predecessors(node)) + list(self.graph.successors(node))
        if lookup=='p': # predecessors
            return list(self.graph.predecessors(node))
        if lookup=='s': # successors
            return list(self.graph.successors(node))
        
    def reset(self, start=None):
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
