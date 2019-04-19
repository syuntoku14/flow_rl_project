from common.evaluate import make_vis_env, test_env
import argparse, json

from flow.multiagent_envs import MultiWaveAttenuationMergePOEnv
from flow.scenarios import MergeScenario
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from ray.tune import run_experiments

import gym, ray
from ray.rllib.agents.ppo import PPOAgent, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

EXAMPLE_USAGE = """
example usage:
    python ppo_runner.py multi_merge
Here the arguments are:
benchmark_name - name of the benchmark to run
num_rollouts - number of rollouts to train across
num_cpus - number of cpus to use for training
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--benchmark_name",
    type=str,
    default='multi_merge',
    help="File path to solution environment.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=60,
    help="The number of rollouts to train over.")

# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=63,
    help="The number of cpus to use.")


if __name__ == "__main__":
    alg_run = "PPO"
    
    args = parser.parse_args()
    benchmark_name = args.benchmark_name
    num_rollouts = args.num_rollouts
    num_cpus = args.num_cpus
    
    gae_lambda = 0.97
    step_size = 5e-4

    ray.init(num_cpus=num_cpus, include_webui=False, ignore_reinit_error=True)

    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params
    horizon = flow_params['env'].horizon

    create_env, env_name = make_create_env(params=flow_params, version=0)

    config = DEFAULT_CONFIG.copy()

    config["num_workers"] = min(num_cpus, num_rollouts) - 1
    config["train_batch_size"] = horizon * num_rollouts
    config["use_gae"] = True
    config["horizon"] = horizon
    config["lambda"] = gae_lambda
    config["lr"] = step_size
    config["vf_clip_param"] = 1e6
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Register as rllib env
    register_env(env_name, create_env)

    exp_tag = {
        "run": alg_run,
        "env": env_name,
        "config": {
            **config
        },
        "checkpoint_freq": 25,
        "max_failures": 999,
        "stop": {
            "training_iteration": 500
        },
        "num_samples": 1,
    }

    trials = run_experiments({
            flow_params["exp_tag"]: exp_tag
        })