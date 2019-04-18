import requests
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from flow.scenarios.figure_eight import Figure8Scenario
from flow.envs.loop.loop_accel import AccelEnv


def plot_and_save(num_iters, rewards, image_path):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('iter %s. reward: %s' % (num_iters[-1], rewards[-1]))
    plt.xlabel('num_iter')
    plt.ylabel('average rewards')
    plt.plot(rewards)
    plt.savefig(image_path)
    plt.show()

def send_line(url, headers, message, image_path):
    # send to line
    payload = {"message" :  message}
    files = {"imageFile": open(image_path, "rb")}
    r = requests.post(url=url ,headers = headers ,params=payload, files=files)

def append_trajectory(trajectory, log_prob, value, state, action, reward, done, device):
    trajectory['log_probs'].append(log_prob)
    trajectory['values'].append(value)
    trajectory['rewards'].append(torch.FloatTensor(reward).unsqueeze(1).to(device))
    trajectory['masks'].append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
    trajectory['states'].append(state)
    trajectory['actions'].append(action)
    return trajectory


def cat_trajectory(trajectory, returns):
    trajectory['log_probs'] = torch.cat(trajectory['log_probs']).detach()
    trajectory['values'] = torch.cat(trajectory['values']).detach()
    trajectory['returns'] = torch.cat(returns).detach()
    trajectory['states'] = torch.cat(trajectory['states'])
    trajectory['actions'] =  torch.cat(trajectory['actions'])
    trajectory['advantages'] = trajectory['returns'] - trajectory['values']
    return trajectory