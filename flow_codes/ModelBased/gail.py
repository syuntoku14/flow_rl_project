import ray, os
import numpy as np
import tensorflow as tf
from ray.tune import Trainable
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.evaluation import PolicyEvaluator, SampleBatch
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.annotations import override
from ray.rllib.agents import Trainer

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params

class GailTrainer(Trainer):
    _name = "GAIL"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PPOPolicyGraph
    
    @override(Trainer)
    def _init(self, config, env_creator):
        self.timestep = 0
        self.local_evaluator = self.make_local_evaluator(
             env_creator, self._policy_graph)        
        self.remote_evaluators = self.make_remote_evaluators(
            env_creator, self._policy_graph, config["num_workers"])
        
    @override(Trainer)    
    def _train(self):
        weights = ray.put(self.local_evaluator.get_weights())
        for e in self.remote_evaluators:
            e.set_weights.remote(weights)       
        batch = SampleBatch.concat_samples(
            ray.get([e.sample.remote() for e in self.remote_evaluators]))
        self.local_evaluator.learn_on_batch(batch)
        return collect_metrics(remote_evaluators=self.remote_evaluators)
