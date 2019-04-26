import os, sys, pickle
import numpy as np
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path = list(set(sys.path))
import matplotlib.pyplot as plt

from results_path import DDPG_PATH, PPO_PATH

from common.evaluate import make_vis_env, test_env

from flow.multiagent_envs import MultiWaveAttenuationMergePOEnv
from flow.scenarios import MergeScenario
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from ray.tune import run_experiments

import gym, ray
from ray.rllib.agents.ppo import PPOAgent, DEFAULT_CONFIG
from ray.rllib.agents.ddpg import DDPGAgent, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

exp_name = '750s_multi_merge'
checkpoint = '50'

results_list = os.listdir('/headless/ray_results/' + exp_name)

benchmark_name = 'multi_merge'
benchmark = __import__(
    "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
flow_params = benchmark.flow_params
horizon = flow_params['env'].horizon
create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

ray.init(num_cpus=1, include_webui=False, ignore_reinit_error=True)


for AGENT_PATH in results_list:
    AGENT = AGENT_PATH[:AGENT_PATH.find('_')]
    PATH = exp_name + '/' + AGENT_PATH
    config_path = '/headless/rl_project/ray_results/' + PATH + '/params.pkl'
    checkpoint_path = '/headless/rl_project/ray_results/' + PATH + '/checkpoint_{}/checkpoint-{}'.format(checkpoint, checkpoint)

    with open(config_path, mode='rb') as f:
        config = pickle.load(f)

    if AGENT == 'PPO':
        agent = PPOAgent(config=config, env=env_name)
    elif AGENT == 'DDPG':
        agent = DDPGAgent(config=config, env=env_name)
        
    agent.restore(checkpoint_path)