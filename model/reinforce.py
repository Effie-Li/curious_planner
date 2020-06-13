import torch
import torch.nn as nn

class Reinforce(nn.Module):
    
    '''
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    '''

    def __init__(self, 
                 in_size, 
                 hid_size, 
                 action_dim,
                 n_layers=1, 
                 fc1=None, 
                 action_head=None, 
                 lr=2e-2, 
                 epsilon=0.05, 
                 gamma=0.9):
        
        super(Reinforce, self).__init__()

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

        # action buffer
        self.saved_log_probs = []
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

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob

    def step(self, state):
        state = torch.from_numpy(np.expand_dims(state,0)).float() # expand state to batch dim
        probs = self.forward(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        if self.test_mode:
            action = m.sample()
        elif np.random.rand() < self.epsilon:
            # sample action randomly
            uni = Categorical(torch.from_numpy(np.tile([1/self.action_dim], self.action_dim)))
            action = uni.sample()
        else:
            action = m.sample()

        # save to action buffer
        self.saved_log_probs.append(m.log_prob(action))

        # the action to take (0,1,2,3)
        return action.item()

    def train(self):
        """
        Training code. Calcultes actor and critic loss and performs backprop.
        """
        R = 0
        saved_log_probs = self.saved_log_probs
        policy_losses = [] # list to save actor (policy) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        for log_prob, R in zip(saved_log_probs, returns):
            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * R)

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses
        loss = torch.stack(policy_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_log_probs[:]