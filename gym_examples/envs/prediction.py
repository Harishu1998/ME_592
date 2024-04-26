import torch
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(self, state_dim, act_dim, model_size=[256, 256]):
        super(Dynamics, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.model_size = model_size
        self.fc1 = nn.Linear(state_dim + act_dim, model_size[0])
        self.fc2 = nn.Linear(model_size[0], model_size[1])
        self.fc3 = nn.Linear(model_size[1], state_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        return self.forward(x)