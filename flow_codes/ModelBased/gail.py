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

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params

logger = logging.getLogger(__name__)

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
        
        self.standardize_fields = ["advantages"]
        self.sample_batch_size = config["sample_batch_size"]
        self.num_envs_per_worker = config["num_envs_per_worker"]
        self.train_batch_size = config["train_batch_size"]
        self.num_sgd_iter = config["num_sgd_iter"]
        self.sgd_minibatch_size = config["sgd_minibatch_size"]
        
        self.policies = dict(
            self.local_evaluator.foreach_trainable_policy(lambda p, i: (i, p)))
        
    @override(Trainer)    
    def _train(self):
        weights = ray.put(self.local_evaluator.get_weights())
        for e in self.remote_evaluators:
            e.set_weights.remote(weights)       
        
        # collect samples
        samples = []
        while sum(s.count for s in samples) < self.train_batch_size:
            samples.extend(
                ray.get([
                    e.sample.remote() for e in self.remote_evaluators
                ]))
        samples = SampleBatch.concat_samples(samples)
        samples.shuffle()
        
        # training
        for i in range(self.num_sgd_iter):
            fetches = self.local_evaluator.learn_on_batch(samples)
            
        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]["kl"])
            else:
                logger.debug(
                    "No data for {}, not updating kl".format(pi_id))
        self.local_evaluator.foreach_trainable_policy(update)
        res = collect_metrics(self.local_evaluator, self.remote_evaluators)
        return res


"""
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
        
        self.optimizer = LocalMultiGPUOptimizer(
            self.local_evaluator,
            self.remote_evaluators,
            sgd_batch_size=config["sgd_minibatch_size"],
            num_sgd_iter=config["num_sgd_iter"],
            num_gpus=config["num_gpus"],
            sample_batch_size=config["sample_batch_size"],
            num_envs_per_worker=config["num_envs_per_worker"],
            train_batch_size=config["train_batch_size"],
            standardize_fields=["advantages"],
            straggler_mitigation=config["straggler_mitigation"])

    @override(Trainer)    
    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        fetches = self.optimizer.step()
        if "kl" in fetches:
            # single-agent
            self.local_evaluator.for_policy(
                lambda pi: pi.update_kl(fetches["kl"]))
        else:

            def update(pi, pi_id):
                if pi_id in fetches:
                    pi.update_kl(fetches[pi_id]["kl"])
                else:
                    logger.debug(
                        "No data for {}, not updating kl".format(pi_id))

            # multi-agent
            self.local_evaluator.foreach_trainable_policy(update)
        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=self.optimizer.num_steps_sampled - prev_steps,
            info=res.get("info", {}))

        # Warn about bad clipping configs
        if self.config["vf_clip_param"] <= 0:
            rew_scale = float("inf")
        elif res["policy_reward_mean"]:
            rew_scale = 0  # punt on handling multiagent case
        else:
            rew_scale = round(
                abs(res["episode_reward_mean"]) / self.config["vf_clip_param"],
                0)
        return res
"""