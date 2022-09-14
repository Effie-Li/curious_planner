import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    
    """
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    implements both actor and critic in one model
    """
    def __init__(self, 
                 in_size, 
                 hid_size, 
                 action_dim,
                 n_layers=1, 
                 fc1=None, 
                 action_head=None, 
                 value_head=None,
                 lr=2e-2, 
                 epsilon=0.05, 
                 gamma=0.9):
        
        super(ActorCritic, self).__init__()

        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_dim = action_dim

        self.fc1 = nn.Sequential(*[nn.Linear(in_size, hid_size), 
                                   nn.ReLU()])
        if fc1 is not None:
            with torch.no_grad():
                self.fc1.load_state_dict(fc1.state_dict())
        
        # self.fc = None
        # if n_layers > 1:
        #     self.fc = []
        #     for i in list(range(n_layers))[1:]:
        #         self.fc.append(nn.Linear(hid_size, hid_size))
        #         self.fc.append(nn.ReLU())
        #     self.fc = nn.Sequential(*self.fc)

        # actor's layer
        self.action_head = nn.Linear(hid_size, self.action_dim)
        if action_head is not None:
            with torch.no_grad():
                self.action_head.load_state_dict(action_head.state_dict())

        # critic's layer
        self.value_head = nn.Linear(hid_size, 1)
        if value_head is not None:
            with torch.no_grad():
                self.value_head.load_state_dict(value_head.state_dict())

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
        self.test_mode = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.fc1(x)
        # if self.fc is not None:
        #     x = self.fc(x)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        # print(self.action_head(x))
        action_prob = nn.functional.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

    def step(self, state):
        state = torch.from_numpy(np.expand_dims(state,0)).float() # expand state to batch dim
        probs, state_value = self.forward(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        if self.test_mode:
            # strictly deterministic
            action = torch.argmax(probs, dim=-1)
            # action = m.sample()
        elif np.random.rand() < self.epsilon:
            # sample action randomly
            uni = Categorical(torch.from_numpy(np.tile([1/self.action_dim], self.action_dim)))
            action = uni.sample()
            # save to action buffer
            self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        else:
            action = m.sample()
            # save to action buffer
            self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (0,1,2,3)
        return action.item()

    def train(self):
        """
        Training code. Calcultes actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        self.reset_buffer()
        
        return loss

    def reset_buffer(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
