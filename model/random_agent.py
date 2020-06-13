import numpy as np

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        
    def step(self, observation):
        return np.random.randint(self.action_dim)