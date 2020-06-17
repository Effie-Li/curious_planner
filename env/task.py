import torch
import torch.nn as nn
import numpy as np

class TraversalTask():
    
    def __init__(self, 
                 world, 
                 start=None, 
                 goal=None,
                 min_dist=2,
                 max_dist=None,
                 change_start_on_reset=False, 
                 change_goal_on_reset=False,
                 goal_conditioned_obs=True,
                 use_onehot_obs=True, 
                 permute_state_labels=False, 
                 reward='sparse'):
        
        '''
        :param change_start_on_reset: given start will be ignored
        :param change_goal_on_reset: given goal will be ignored
        '''
        
        self.world = world
        self.start = start
        self.goal = goal
        self.min_dist = min_dist # if task difficulty should be lower-bounded by n-step distance
        self.max_dist = max_dist
        self.change_start_on_reset = change_start_on_reset
        self.change_goal_on_reset = change_goal_on_reset
        self.goal_conditioned_obs = goal_conditioned_obs
        self.use_onehot_obs = use_onehot_obs
        self.permute_state_labels = permute_state_labels # TODO: implement permute_state_labels
        self.reward = reward
        self.action_dim = world.action_dim
        
        self._set_start_goal()
    
    def _set_start_goal(self):
        # TODO: improve algorithm
        # currently this is done by surveying all possible nodes in the network
        # as we're focusing on small network for now
        if (self.change_start_on_reset) and (not self.change_goal_on_reset):
            self.start = self._sample_node_at_dist(self.goal, self.min_dist, self.max_dist, direction='to')

        if (not self.change_start_on_reset) and (self.change_goal_on_reset):
            self.goal = self._sample_node_at_dist(self.start, self.min_dist, self.max_dist, direction='from')

        if (self.change_start_on_reset) and (self.change_goal_on_reset):
            self.start = self.world.reset()
            self.goal = self._sample_node_at_dist(self.start, self.min_dist, self.max_dist, direction='from')

        return self.start, self.goal

    def _sample_node_at_dist(self, node, min_dist, max_dist, direction):
        # sample a node at certain distance from the given node
        # direction: out, distance is based on given node --> target node
        #            in, distance is based on target node --> given node
        # TODO: what if there are no valid nodes?
        valid_nodes = []
        for x in self.world.nodes:
            if direction=='from':
                shortest_path = self.world.shortest_path(node, x)
            elif direction=='to':
                shortest_path = self.world.shortest_path(x, node)
            check1 = True if min_dist is None else (shortest_path['n_step']>=min_dist)
            check2 = True if max_dist is None else (shortest_path['n_step']<=max_dist)
            if check1 and check2:
                valid_nodes.append(x)
        target = np.random.choice(valid_nodes)
        return target
        
    def _choose_exclude(self, nodes, exclude):
        # randomly choose from nodes excluding the given states
        valid_nodes = np.array([n for n in nodes if n not in exclude])
        return np.random.choice(valid_nodes)
    
    def _to_onehot(self, obs):
        if self.goal_conditioned_obs:
            s = obs[0]
            g = obs[1]
            n = len(self.world.nodes)
            obs = torch.cat((nn.functional.one_hot(torch.tensor(s), n),
                             nn.functional.one_hot(torch.tensor(g), n)),
                            dim=0).type('torch.FloatTensor')
        else:
            n = len(self.world.nodes)
            obs = nn.functional.one_hot(torch.tensor(obs), n).type('torch.FloatTensor')
        return obs
    
    def reset(self):
        self._set_start_goal()
        self.world.reset(self.start)
        if self.goal_conditioned_obs:
            obs = [self.start, self.goal]
        else:
            obs = self.start
        if self.use_onehot_obs:
            obs = self._to_onehot(obs)
        return obs
    
    def step(self, action):
        
        ns = self.world.step(action)
        if self.goal_conditioned_obs:
            obs = [ns, self.goal]
        else:
            obs = ns
        if self.use_onehot_obs:
            obs = self._to_onehot(obs)
        done = ns == self.goal
        if self.reward == 'sparse':
            r = 1.0 if done else 0.0
        elif self.reward == '-1':
            r = -1.0
        
        return obs, r, done
