import torch
import torch.nn as nn

class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        # Capa de entrada de 7 neuronas (7 entradas)
        self.fc1 = nn.Linear(7, 8)
        # Capa oculta de 8 neuronas
        # Capa de salida de 3 neuronas (3 salidas)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        # Funcion de activacion ReLU
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
