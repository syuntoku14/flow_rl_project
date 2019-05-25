import gym, pickle, argparse, json
import numpy as np
from itertools import product
from copy import deepcopy
import tensorflow as tf
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.evaluation import PolicyEvaluator, SampleBatch
from ray.rllib.evaluation.metrics import collect_metrics
from ray.tune.registry import register_env

from flow.multiagent_envs import MultiWaveAttenuationPOEnv
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from gail import GailTrainer

parser = argparse.ArgumentParser()

# required input parameters
parser.add_argument("--exp_name", type=str)

# optional input parameters
parser.add_argument("--benchmark_name", type=str, default="multi_merge")
parser.add_argument("--num_cpus", type=int, default=3)
parser.add_argument("--num_rollouts", type=int, default=2)
parser.add_argument("--num_iter", type=int, default=3)


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["cost1"] = []
    episode.user_data["cost2"] = []
    episode.user_data["mean_vel"] = []
    episode.user_data["outflow"] = []


def on_episode_step(info):
    episode = info["episode"]
    agent_ids = episode.agent_rewards.keys()
    infos = [episode.last_info_for(id_[0]) for id_ in agent_ids]
    cost1, cost2, mean_vel, outflow = 0, 0, 0, 0
    if len(infos) != 0:
        cost1 = np.mean([info['cost1'] for info in infos])
        cost2 = np.mean([info['cost2'] for info in infos])
        mean_vel = np.mean([info['mean_vel'] for info in infos])
        outflow = np.mean([info['outflow'] for info in infos])
    episode.user_data["cost1"].append(cost1)
    episode.user_data["cost2"].append(cost2)
    episode.user_data["mean_vel"].append(mean_vel)
    episode.user_data["outflow"].append(outflow)
    
    
def on_episode_end(info):
    episode = info["episode"]
    cost1 = np.sum(episode.user_data["cost1"])
    cost2 = np.sum(episode.user_data["cost2"])
    mean_vel = np.mean(episode.user_data["mean_vel"])
    outflow = np.mean(episode.user_data["outflow"][-500:])  # 1/3 of the whole steps
    episode.custom_metrics["cost1"] = cost1
    episode.custom_metrics["cost2"] = cost2
    episode.custom_metrics["system_level_velocity"] = mean_vel
    episode.custom_metrics["outflow_rate"] = outflow
        
        
def main():
    args = parser.parse_args()
    num_cpus = args.num_cpus
    num_rollouts = args.num_rollouts
    num_iter = args.num_iter
    benchmark_name = args.benchmark_name
    exp_name = args.exp_name
    gae_lambda = 0.97
    step_size = 5e-4

    ray.init(num_cpus=num_cpus, logging_level=50, ignore_reinit_error=True)
    
    # set the config
    benchmark = __import__(
                "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.buffered_obs_flow_params    
    horizon = flow_params["env"].horizon
    
    config = deepcopy(DEFAULT_CONFIG)
    config["num_workers"] = min(num_cpus, num_rollouts)
    config["train_batch_size"] = horizon * num_rollouts
    config["sample_batch_size"] = horizon / 2
    config["use_gae"] = True
    config["horizon"] = horizon
    config["lambda"] = gae_lambda
    config["lr"] = step_size
    config["vf_clip_param"] = 1e6
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["model"]["fcnet_hiddens"] = [128, 64, 32]
    config["observation_filter"] = "NoFilter"
    config["entropy_coeff"] = 0.0
    
    config['callbacks']['on_episode_start'] = ray.tune.function(on_episode_start)
    config['callbacks']['on_episode_step'] = ray.tune.function(on_episode_step)
    config['callbacks']['on_episode_end'] = ray.tune.function(on_episode_end)

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    
    # register environment
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)
    
    # set policy_graph to config
    env = create_env()
    default_policy = (PPOPolicyGraph, env.observation_space, env.action_space, {})
    policy_graph = {"default_policy": default_policy}
    config["multiagent"] = {
            'policy_graphs': policy_graph,
            'policy_mapping_fn': tune.function(lambda agent_id: "default_policy")
        }
    
    tune.run_experiments({
        exp_name: {
            "run": GailTrainer,
            "env": env_name,
            "checkpoint_freq": 25,
            "max_failures": 999,
            "num_samples": 5,
            "stop": {
                "training_iteration": num_iter
            },
            "config": config
        }   
    })

    
if __name__ == "__main__":
    main()
