from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import sys

import chainer
from chainer import optimizers
import gym
from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents.ddpg_ma import DDPG_MA
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import policy
from chainerrl import q_functions
from chainerrl import replay_buffer

from flow.multiagent_envs import MultiWaveAttenuationMergePOEnvMeanRew
from flow.scenarios import MergeScenario
from flow.utils.registry import make_create_env


def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    # simulation parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)    
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--num-envs', type=int, default=62)
    
    # model parameter
    parser.add_argument('--steps', type=int, default=10 ** 4)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--minibatch-size', type=int, default=500)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--n-hidden-channels', type=int, default=300)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--replay-start-size', type=int, default=2000)
    parser.add_argument('--n-update-times', type=int, default=1)
    parser.add_argument('--target-update-interval',
                        type=int, default=15)
    parser.add_argument('--target-update-method',
                        type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--use-bn', action='store_true', default=False)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)

    args = parser.parse_args()
    
    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    
    # make create_env function for multi_merge
    benchmark_name = 'multi_merge'
    benchmark  = __import__(
    "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.mean_rew_flow_params
    HORIZON = flow_params['env'].horizon
    create_env, env_name = make_create_env(params=flow_params, version=0)
    
    def make_env(create_env, test):
        def _thunk():
            env = create_env()
            if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
                env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
            return env
        return _thunk

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [make_env(create_env, test) for i in range(args.num_envs)])

    sample_env = create_env()
    obs_size = np.asarray(sample_env.observation_space.shape).prod()
    action_space = sample_env.action_space

    action_size = np.asarray(action_space.shape).prod()
    if args.use_bn:
        q_func = q_functions.FCBNLateActionSAQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            normalize_input=True)
        pi = policy.FCBNDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True,
            normalize_input=True)
    else:
        q_func = q_functions.FCSAQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
        pi = policy.FCDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True)
    model = DDPGModel(q_func=q_func, policy=pi)
    opt_a = optimizers.Adam(alpha=args.actor_lr)
    opt_c = optimizers.Adam(alpha=args.critic_lr)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    ou_sigma = (action_space.high - action_space.low) * 0.2
    explorer = explorers.AdditiveOU(sigma=ou_sigma)
    agent = DDPG_MA(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_method=args.target_update_method,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 soft_update_tau=args.soft_update_tau,
                 n_times_update=args.n_update_times,
                 gpu=args.gpu, minibatch_size=args.minibatch_size)

    if len(args.load) > 0:
        agent.load(args.load)

    if args.demo:
        env = make_batch_env(test=True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        env = make_batch_env(test=False)
        
        experiments.train_agent_batch_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_env=env, eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir)

if __name__ == '__main__':
    main()
