import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from env.remote_vector_env import dict_to_array

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, fcnet_hiddens):
        super().__init__()
        last_layer_size = num_inputs
        self.layers = []
        
        for size in fcnet_hiddens:
            self.layers.append(nn.Linear(last_layer_size, size))
            self.layers.append(nn.ReLU())
            last_layer_size = size
        
    
class Actor(Model):
    def __init__(self, num_inputs, num_outputs, fcnet_hiddens, std=0.0):
        super().__init__(num_inputs, num_outputs, fcnet_hiddens)
        self.layers.append(nn.Linear(fcnet_hiddens[-1], num_outputs))
        self.actor = nn.Sequential(*self.layers)
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)
        
    def forward(self, x):
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist
        
    def select_action(self, x):
        action_dict = {}
        log_prob_dict = {}
        
        with torch.no_grad():
            obss, id_list = dict_to_array(x)    

            x = torch.tensor(obss).float()
            dist = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.cpu().numpy()

            for id_, a_, log_p_ in zip(id_list, action, log_prob):
                env_id, agent_id = id_
                if not env_id in action_dict:
                    action_dict.update({env_id: {}})
                    log_prob_dict.update({env_id: {}})
                action_dict[env_id].update({agent_id: a_})       
                log_prob_dict[env_id].update({agent_id: log_p_})
        return action_dict, log_prob_dict
    
    
        
class Critic(Model):
    def __init__(self, num_inputs, num_outputs, fcnet_hiddens):
        super().__init__(num_inputs, num_outputs, fcnet_hiddens)
        self.layers.append(nn.Linear(fcnet_hiddens[-1], 1))
        self.critic = nn.Sequential(*self.layers)
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        return value
    