import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=0)

        return x
