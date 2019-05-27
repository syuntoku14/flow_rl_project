import ray, os, logging
import numpy as np
import tensorflow as tf
from ray.tune import Trainable
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.evaluation import PolicyEvaluator, SampleBatch
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.optimizers.rollout import collect_samples
from ray.rllib.optimizers import LocalMultiGPUOptimizer
from ray.rllib.utils.annotations import override
from ray.rllib.agents import Trainer
from ray.rllib.offline.json_reader import JsonReader
from ray.tune.logger import pretty_print

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params

logger = logging.getLogger(__name__)

class MEIRTrainer(Trainer):
    _allow_unknown_configs = True
    _name = "MEIR"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PPOPolicyGraph
    
    @override(Trainer)
    def _init(self, config, env_name):
        self.local_evaluator = self.make_local_evaluator(
             env_name, self._policy_graph)        
        self.remote_evaluators = self.make_remote_evaluators(
            env_name, self._policy_graph, config["num_workers"])
        
        self.train_batch_size = config["train_batch_size"]
        self.num_sgd_iter = config["num_sgd_iter"]
        self.num_train = config["num_train"]
        self.expert_path = config["expert_path"]
        self.theta_lr = config["theta_lr"]
        
        expert_reader = JsonReader(self.expert_path)
        self.expert_samples = expert_reader.next()
        self.expert_features = self.calculate_expected_feature(self.expert_samples)
        self.theta = np.random.uniform(size=self.expert_features.shape)
        
    def sample(self, sample_size):
        # set local weights to remote
        weights = ray.put(self.local_evaluator.get_weights())
        for e in self.remote_evaluators:
            e.set_weights.remote(weights)
            
        samples = []
        while sum(s.count for s in samples) < sample_size:
            samples.extend(
                ray.get([
                    e.sample.remote() for e in self.remote_evaluators
                ]))
        samples = SampleBatch.concat_samples(samples)
        return samples
    
    def calculate_expected_feature(self, samples):
        features = np.mean(samples["obs"], axis=0)
        return features
    
    def train_policy_by_samples(self, samples):
        # train policy by given samples
        for i in range(self.num_sgd_iter):
            fetches = self.local_evaluator.learn_on_batch(samples)
            
        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]["kl"])
            else:
                logger.debug(
                    "No data for {}, not updating kl".format(pi_id))
        self.local_evaluator.foreach_trainable_policy(update)       
        
    def set_new_rewards(self, samples):
        samples["rewards"] = samples["obs"].dot(self.theta.T)
        policy = self.get_policy()
        return policy.postprocess_trajectory(samples)
    
    def update_theta(self, samples, learning_rate=0.01):
        # update and return the difference norm
        features = self.calculate_expected_feature(samples)
        update = self.expert_features - features
        self.theta += learning_rate * update
        return np.linalg.norm(self.expert_features - features)
    
    @override(Trainer)    
    def _train(self):
        
        # optimize policy under estimated reward
        for train_iter in range(self.num_train):
            # collect samples with new reward fnc
            samples = self.sample(self.train_batch_size)
            samples = self.set_new_rewards(samples)
            samples.shuffle()

            # train local based on samples
            self.train_policy_by_samples(samples)
            res = collect_metrics(self.local_evaluator, self.remote_evaluators)
            pretty_print(res)
        
        samples = self.sample(self.train_batch_size) 
        norm = self.update_theta(samples, self.theta_lr)
        
        res["custom_metrics"]["theta_norm"] = norm
        return res

    @override(Trainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["theta"] = self.theta
        return state
    
    @override(Trainer)
    def __setstate__(self, state):
        super().__setstate__(state)
        if "theta" in state:
            self.theta = state["theta"]
