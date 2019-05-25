import ray, os
import numpy as np
import tensorflow as tf
from ray.tune import Trainable
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.evaluation import PolicyEvaluator, SampleBatch
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.optimizers.rollout import collect_samples
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
        self.local_evaluator = self.make_local_evaluator(
             env_creator, self._policy_graph)        
        self.remote_evaluators = self.make_remote_evaluators(
            env_creator, self._policy_graph, config["num_workers"])
        
        self.sample_batch_size = config["sample_batch_size"]
        self.num_envs_per_worker = config["num_envs_per_worker"]
        self.train_batch_size = config["train_batch_size"]
        self.num_sgd_iter = config["num_sgd_iter"]
        self.sgd_minibatch_size = config["sgd_minibatch_size"]
        
    @override(Trainer)    
    def _train(self):
        weights = ray.put(self.local_evaluator.get_weights())
        for e in self.remote_evaluators:
            e.set_weights.remote(weights)       
        
        # collect samples
        samples = collect_samples(
            self.remote_evaluators, self.sample_batch_size,
            self.num_envs_per_worker, self.train_batch_size)
        
        samples.shuffle()
        
        # training
        for _ in range(self.num_sgd_iter):
            for i in range(0, samples.count, self.sgd_minibatch_size):
                minibatch = samples.slice(i, i+self.sgd_minibatch_size)
                self.local_evaluator.learn_on_batch(minibatch)
        return collect_metrics(remote_evaluators=self.remote_evaluators)
