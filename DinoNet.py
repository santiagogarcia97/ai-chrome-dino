import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.fc1 = nn.Linear(7, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
