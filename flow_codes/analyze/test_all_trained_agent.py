import os, sys, pickle, argparse
from tqdm import tqdm
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

EXAMPLE_USAGE = """
example usage:
    python test_all_trained_agent.py --exp_name 750s_multi_merge --checkpoint 50
Here the arguments are:
    exp_name: name of the experiment
    checkpoint: checkpoint to evaluate
    try_6_times: when specified, evaluate 6 times for each agent
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--exp_name", type=str, help="experiment name")

parser.add_argument(
    "--checkpoint", type=str, help="checkpoint of the model")

parser.add_argument(
    "--try_6_times", action='store_true')

args = parser.parse_args()
exp_name = args.exp_name
checkpoint = args.checkpoint
try_6_times = args.try_6_times

print("Evaluation starts")
print("exp_name: {} \ncheckpoint: {} \ntry_6_times: {}".format(exp_name, checkpoint, try_6_times))

results_list = os.listdir('/headless/ray_results/' + exp_name)

benchmark_name = 'multi_merge'
benchmark = __import__(
    "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
flow_params = benchmark.flow_params
horizon = flow_params['env'].horizon
create_env, env_name = make_create_env(params=flow_params, version=0)

ray.init(num_cpus=1, include_webui=False, ignore_reinit_error=True)

register_env(env_name, create_env)

for AGENT_PATH in tqdm(results_list, desc='Results'):
    print('AGENT_PATH: {}'.format(AGENT_PATH))
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
        
    try:
        agent.restore(checkpoint_path)
    except:
        print("{} \n checkpoint doesn't exist".format(PATH + '/checkpoint_{}/checkpoint-{}'.format(checkpoint, checkpoint)))
        pass
    
    env = create_env()
    # calculate the space-time velocity map
    left_length = env.k.scenario.edge_length('left')
    car_length = 5.0
    scale = 10
    vel_lists = []
    vel_map_lists = []
    outflow_lists = []

    num_iter = 6 if try_6_times else 1
    
    for _ in tqdm(range(num_iter), desc='Trials'):
        state = env.reset()
        vel = []
        outflow = []
        for i in tqdm(range(env.env_params.horizon), desc='env step'):
            # record the mean velocity
            v = np.mean(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))
            vel.append(v)
            
            # record the velocity map
            ids = env.k.vehicle.get_ids()
            vel_map_ = np.zeros(int(left_length)*scale)
            for id_ in ids:
                pos_ = np.round(env.k.vehicle.get_position(id_), decimals=1)
                vel_ = env.k.vehicle.get_speed(id_)
                pos_bottom = max(0, int((pos_-car_length/2.0)*scale))
                pos_top = min(int(left_length)*scale, int((pos_+car_length/2.0)*scale))
                vel_map_[pos_bottom:pos_top] = vel_            

            # step the simulation
            rl_ids = env.k.vehicle.get_rl_ids()
            actions = {}
            if AGENT != 'HUMAN':
                for id_ in rl_ids:
                    action = agent.compute_action(state[id_])
                    actions.update({id_: action})
            state, r, _, _ = env.step(actions)
            outflow.append(env.k.vehicle.get_outflow_rate(600)) # measured by one min

            if i == 0:
                vel_map_list = vel_map_
            else:
                vel_map_list = np.vstack((vel_map_list, vel_map_))

        outflow_lists.append(outflow[-500:])
        vel_lists.append(vel)
        vel_map_list[vel_map_list==0.0] = np.nan
        vel_map_lists.append(vel_map_list)
        
    vel_fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range(num_iter):
        ax.plot(vel_lists[i])
    title = AGENT + ' multi-agent mean velocity'
    title = title + ' \n mean velocity {:3f}'.format(np.mean(vel_lists))
    title = title + ' \n mean outflow rate {:3f}'.format(np.mean(outflow_lists))
    ax.set_ylabel('system level mean velocity (m/s)')
    ax.set_xlabel('step')
    vel_fig.suptitle(title)
    plt.subplots_adjust(top=0.8)
    
    
    nrows = 2 if try_6_times else 1
    ncols = 3 if try_6_times else 1
    space_fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    space_fig.tight_layout()

    for i in range(num_iter):
        # plt.subplot(2, 3, i+1)
        x = np.arange(int(env.env_params.horizon))
        y = np.arange(0, int(left_length), step=0.1)
        xx, yy = np.meshgrid(x, y)
        try:
            im = axes[i%2, i%3].pcolormesh(xx, yy, vel_map_lists[i].T)
            if i % 3 == 0:
                axes[i%2, i%3].set_ylabel('Position (m)')
            if int(i / 3.0) > 0:
                axes[int(i/3), int(i/3)].set_xlabel('step')
        except:
            im = axes.pcolormesh(xx, yy, vel_map_lists[i].T)
            if i % 3 == 0:
                axes.set_ylabel('Position (m)')
            if int(i / 3.0) > 0:
                axes.set_xlabel('step')

    try:
        clb = space_fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    except:
        clb = space_fig.colorbar(im, ax=axes, shrink=0.95)

    clb.set_clim(0, 30)
    clb.set_label('Velocity (m/s)')
    title = AGENT + ' Space-Time Diagram of 600 meter merge road'
    space_fig.suptitle(title)   
    plt.subplots_adjust(top=0.9, right=0.8)
    
    # save figures
    filename = PATH[PATH.find('/') + 1:].replace(' ', '_') + '.png'
    vel_dir = '../result/MultiMerge/' + AGENT + '/' + exp_name + '/mean_velocity/'
    space_dir = '../result/MultiMerge/' + AGENT + '/' + exp_name + '/space_time_diagram/'
    os.makedirs(vel_dir, exist_ok=True)
    os.makedirs(space_dir, exist_ok=True)

    vel_fig.savefig(vel_dir + filename)
    space_fig.savefig(space_dir + filename)
