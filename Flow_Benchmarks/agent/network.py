import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, fcnet_hiddens, std=0.0):
        super(ActorCritic, self).__init__()
        last_layer_size = num_inputs
        layers = []
        
        for size in fcnet_hiddens:
            layers.append(nn.Linear(last_layer_size, size))
            layers.append(nn.ReLU())
            last_layer_size = size
            
        layers.append(nn.Linear(fcnet_hiddens[-1], num_outputs))
        
        self.critic = nn.Sequential(*layers)
        self.actor = nn.Sequential(*layers)
        
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value