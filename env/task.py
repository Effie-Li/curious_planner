import torch
import torch.nn as nn
import numpy as np

class TraversalTask():
    
    def __init__(self, 
                 world, 
                 start=None, 
                 goal=None,
                 fix_dist = None,
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
        self.fix_dist = fix_dist
        self.change_start_on_reset = change_start_on_reset
        self.change_goal_on_reset = change_goal_on_reset
        self.goal_conditioned_obs = goal_conditioned_obs
        self.use_onehot_obs = use_onehot_obs
        self.permute_state_labels = permute_state_labels # TODO: implement permute_state_labels
        self.reward = reward
        self.action_dim = world.action_dim
        
        self._set_start_goal()
    
    def _set_start_goal(self):
        if self.change_start_on_reset:
            self.start = self._choose_exclude(self.world.nodes, self.world.get_neighbors(self.goal))
        if self.change_goal_on_reset:
            if self.fix_dist is not None:
                self.goal = self._n_step_goal(self, self.start, self.fix_dist)
            else:
                self.goal = self._choose_exclude(self.world.nodes, self.world.get_neighbors(self.start))
        return self.start, self.goal
    
    def _n_step_goal(self, start, n_steps):
        # force n-step task
        for _ in range(100):
            goal = self._choose_exclude(self.world.nodes, self.world.get_neighbors(self.start))
            n, _, _ = self.world.shortest_path(self.start, goal)
            if (n==n_steps):
                break
        return goal
        
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
