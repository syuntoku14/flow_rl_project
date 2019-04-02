from flow.scenarios.figure_eight import Figure8Scenario
from flow.envs.loop.loop_accel import AccelEnv

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