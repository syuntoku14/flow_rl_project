import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from flow.scenarios.figure_eight import Figure8Scenario
from flow.envs.loop.loop_accel import AccelEnv
import ray

def make_vis_env(benchmark_name):
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params
    env_params = flow_params['env']
    sim_params = flow_params['sim']
    sim_params.render = True
    
    scenario = Figure8Scenario(
        flow_params['exp_tag'],
        flow_params['veh'],
        flow_params['net'],
        initial_config=flow_params['initial'])
    env = AccelEnv(env_params, sim_params, scenario)
    return env
   
    
def test_env(env, device, model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


@ray.remote
class PolicyEvaluator:
    def __init__(self, create_env):
        self.create_env = create_env
        self.env = create_env()
        
    def test_env(self, device_id, model_id):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device_id)
            dist, _ = model_id(state)
            next_state, reward, done, _ = self.env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            total_reward += reward
        return total_reward