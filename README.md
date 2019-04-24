# flow_rl_project

The goal of this project is to develop an multi-agent for multi-merge env in "flow".

## Setup

1. docker pull syuntoku/flowtouch:latest
2. git clone git@github.com:syuntoku14/flow_rl_project.git
3. git clone git@github.com:syuntoku14/flow.git

When modifying the repository, please create your branch at first.

4. modify "syuntoku14" in flow_rl_project/bash_codes/run_flow.bash as your username.
5. modify the port number in flow_rl_project/bash_codes/run_jupyter.bash

## How to use

### Go inside the docker container
Run
```
./flow_rl_project/bash_codes/run_flow.bash
```

### Run jupyter notebook
Run 
```
./rl_project/bash_code/run_jupyter.bash
```
inside the container

Example codes are in ./rl_project/flow_codes/test

### Train an agent by rllib

The codes are written in flow/flow/benchmarks/rllib

For example, if you want to train ppo with 64 cpus, 

```
python ppo_runner.py --benchmark_name multi_merge --num_cpus 64
```

### Parameter tuning

If you want to tune the parameter, just modify the runner.py codes.

For example, 
```
config["parameter_noise"] = ray.tune.grid_search([True, False])
```

in ddpg_runner.py 

If you want to modify the scale of the reward, modify flow/flow/multiagent_envs/merge.py
