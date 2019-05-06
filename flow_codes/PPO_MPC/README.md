# Multi Merge

## core/multi_agent.py/MultiAgent:

* Collect samples using remote_vector_env
* generate batch used for training

## utils/remote_vector_env:

* RemoteVectorEnv: obs and rewards will be returned as dictionary. ex) {env_id: {agent_id: obs_arr}} 
* dict_to_array: convert dictionary to list of array.

## examples/multi-merge.ipynb

* where you test the code