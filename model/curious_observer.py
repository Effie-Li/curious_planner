import torch
import torch.nn as nn

class CuriousObserver(nn.Module):

    def __init__(self, 
                 in_size, 
                 out_size, 
                 hid_size,
                 nonlinearity='relu',
                 lr=2e-2):

        # TODO: allow pass in a network

        super().__init__()

        if nonlinearity is None:
            self.fc1 = nn.Linear(in_size, hid_size)
        elif nonlinearity=='relu':
            self.fc1 = nn.Sequential(*[nn.Linear(in_size, hid_size),
                                       nn.ReLU()])
        self.fc2 = nn.Linear(hid_size, out_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def train(self, X, y, loss_fn=nn.CrossEntropyLoss()):
        yhat = self.forward(X)
        self.optimizer.zero_grad()
        loss = loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()

        return loss

class IntuitivePathObserver(nn.Module):

    def __init__(self, 
                 in_size_1, 
                 in_size_2,
                 out_size, 
                 hid_size,
                 nonlinearity='relu',
                 lr=2e-2):

        super().__init__()
        self.nonlinearity = nonlinearity

        self.fc1_1 = nn.Linear(in_size_1, hid_size)
        self.fc1_2 = nn.Linear(in_size_2, hid_size)

        self.fc1 = nn.ReLU()

        self.fc2 = nn.Linear(hid_size, out_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x1, x2):
        return self.fc2(self.fc1(self.fc1_1(x1)+self.fc1_2(x2)))

    def train(self, X1, X2, y, loss_fn=nn.CrossEntropyLoss()):
        yhat = self.forward(X1, X2)
        self.optimizer.zero_grad()
        loss = loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()

        return loss
