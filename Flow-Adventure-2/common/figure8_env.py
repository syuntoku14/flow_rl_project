from flow.controllers import RLController, IDMController, StaticLaneChanger, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams, InitialConfig
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.figure_eight import Figure8Scenario, ADDITIONAL_NET_PARAMS


def gen_figure8_env(HORIZON=1500, sim_step=0.1, render=False):
    # SumoParams
    sim_params = SumoParams(sim_step=sim_step, render=render)


    # Vehicles Setting
    vehicles = VehicleParams()

    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=1)

    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        initial_speed=0,
        num_vehicles=14)


    # Additional Env params
    HORIZON = HORIZON

    additional_env_params = {
        "target_velocity": 20,
        "max_accel": 3,
        "max_decel": 3,
        "sort_vehicles": False
    }
    env_params = EnvParams(
        horizon=HORIZON, additional_params=additional_env_params)


    # Additional Net params
    additional_net_params = {
        "radius_ring": 30,
        "lanes": 1,
        "speed_limit": 30,
        "resolution": 40
    }
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)


    ## Initial config
    initial_config = InitialConfig(spacing="uniform")


    ## Scenario
    exp_tag = "figure-eight-control"

    env_name = "figure_eight"

    scenario = Figure8Scenario(
        exp_tag,
        vehicles,
        net_params,
        initial_config=initial_config)


    env = AccelEnv(env_params, sim_params, scenario)
    
    return env