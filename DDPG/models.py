from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, custom_arch = None):
        self.n_states = n_states[0]
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        self.custom_arch = custom_arch

        super(Actor, self).__init__()

        if custom_arch is None:

            self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals, out_features=self.n_hidden1)
            self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
            self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
            self.output = nn.Linear(in_features=self.n_hidden3, out_features=self.n_actions)
        else:
            fc = [nn.Linear(self.n_states + self.n_goals, custom_arch[0])]
            for i in range(len(custom_arch) - 1):
                # assert fc_dim > 0, fc_dim
                fc.append(nn.ReLU(inplace=True))
                # fc.append(nn.Dropout(p=0.5, inplace=False))
                fc.append(nn.Linear(in_features=custom_arch[i], out_features=custom_arch[i+1]))
            self.fc = nn.Sequential(*fc)
            self.output = nn.Linear(custom_arch[-1], self.n_actions)


    def forward(self, x):
        if self.custom_arch is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        else:
            x = F.relu(self.fc(state))
        output = torch.tanh(self.output(x))  # TODO add scale of the action

        return output


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, action_size=1):
        self.n_states = n_states[0]
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        self.action_size = action_size
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)

        return output
