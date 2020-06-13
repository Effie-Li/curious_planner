import torch.nn as nn

class CuriousObserver(nn.Module):

    def __init__(self, in_size, out_size, hid_size):

        super().__init__()

        self.fc1 = nn.Sequential(*[nn.Linear(in_size, hid_size),
                                   nn.ReLU()])
        self.fc2 = nn.Linear(hid_size, out_size)

    def forward(self, x):
        return self.fc2(self.fc1(x))

    # def train(self):
