import gym, pickle, argparse, json, logging
from gym import ObservationWrapper
from copy import deepcopy
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents import Trainer
from ray.rllib.evaluation import PolicyEvaluator, SampleBatch, MultiAgentBatch
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.offline.json_reader import JsonReader
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.multiagent_envs.merge import Discriminator
logger = logging.getLogger(__name__)


class CustomEnvPolicyEvaluator(PolicyEvaluator):
    def set_state_dict(self, state_dict):
        self.env.set_state_dict(state_dict)
        
    def init_discriminator(self, hidden_size):
        self.env.init_discriminator(hidden_size)
     

    
class GAILTrainer(Trainer):
    _allow_unknown_configs = True
    _name = "GAIL"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PPOPolicyGraph
    
    @override(Trainer)
    def _init(self, config, env_name):
        self.train_batch_size = self.config["train_batch_size"]
        self.num_sgd_iter = self.config["num_sgd_iter"]
        
        # load expert trajectory
        self.expert_reader = JsonReader(self.config["expert_path"])
        self.expert_samples = self.expert_reader.next()
               
        # set evaluators
        self.local_evaluator = self.make_local_evaluator(
             env_name, self._policy_graph, self.config)        
        self.remote_evaluators = self.make_remote_evaluators(
            env_name, self._policy_graph, self.config["num_workers"])
       
        # discriminator
        num_inputs = self.local_evaluator.env.observation_space.shape[0]
        num_outputs = self.local_evaluator.env.action_space.shape[0]
        self.discrim_criterion = nn.BCELoss()
        self.discriminator = Discriminator(num_inputs+num_outputs,
                                           config["discrim_hidden_size"])
        self.optimizer_discrim = optim.Adam(self.discriminator.parameters(),
                                            lr=config["lr"])
 
        # share discriminators
        self.local_evaluator.init_discriminator(config["discrim_hidden_size"])
        for e in self.remote_evaluators:
            e.init_discriminator.remote(config["discrim_hidden_size"])
        self.set_state_dict()
            
    def set_state_dict(self):
        state_dict =  self.discriminator.state_dict()
        self.local_evaluator.set_state_dict(state_dict)
        for e in self.remote_evaluators:
            e.set_state_dict.remote(state_dict)  
        
    def get_state_action_from_samples(self, samples):
        state_action = np.hstack((samples["obs"], samples["actions"]))
        state_action = torch.FloatTensor(state_action)
        return state_action

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
    
    def train_policy_by_samples(self, samples):
        # train policy by given samples
        for i in range(self.num_sgd_iter):
            fetches = self.local_evaluator.learn_on_batch(samples)
            
        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]['learner_stats']["kl"])
            else:
                logger.debug(
                    "No data for {}, not updating kl".format(pi_id))
        self.local_evaluator.foreach_trainable_policy(update)       
   
    def train_discriminator_by_state_action(self, state_action, expert_state_action):
        fake = self.discriminator(state_action)
        real = self.discriminator(expert_state_action)       
        self.optimizer_discrim.zero_grad()
        # if perfect, fake == 1, real == 0
        discrim_loss = self.discrim_criterion(fake, torch.ones((state_action.shape[0], 1)).cpu())
        discrim_loss += self.discrim_criterion(real, 
                       torch.zeros((expert_state_action.size(0), 1)).cpu())        
        discrim_loss.backward()
        self.optimizer_discrim.step()
        
        return discrim_loss
        
    @override(Trainer)    
    def _train(self):
        samples = self.sample(self.train_batch_size)
        samples.shuffle()
        self.expert_samples = self.expert_reader.next()
        self.expert_samples.shuffle()
        state_action = self.get_state_action_from_samples(samples)
        expert_state_action = self.get_state_action_from_samples(self.expert_samples)
        
        self.train_policy_by_samples(samples)
        discrim_loss = self.train_discriminator_by_state_action(state_action, expert_state_action)
        
        res = collect_metrics(self.local_evaluator, self.remote_evaluators)
        res["custom_metrics"]["discrim_loss"] =  discrim_loss.data.item()
        pretty_print(res)
        return res

    @override(Trainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["discrim_state_dict"] = self.discriminator.state_dict()
        return state
    
    @override(Trainer)
    def __setstate__(self, state):
        super().__setstate__(state)
        self.discriminator.load_state_dict(state["discrim_state_dict"])

    def make_local_evaluator(self,
                             env_creator,
                             policy_graph,
                             extra_config=None):
        """Convenience method to return configured local evaluator."""

        return self._make_evaluator(
            CustomEnvPolicyEvaluator,
            env_creator,
            policy_graph,
            0,
            merge_dicts(
                # important: allow local tf to use more CPUs for optimization
                merge_dicts(
                    self.config, {
                        "tf_session_args": self.
                        config["local_evaluator_tf_session_args"]
                    }),
                extra_config or {}))        
    
    def make_remote_evaluators(self, env_creator, policy_graph, count):
        """Convenience method to return a number of remote evaluators."""

        remote_args = {
            "num_cpus": self.config["num_cpus_per_worker"],
            "num_gpus": self.config["num_gpus_per_worker"],
            "resources": self.config["custom_resources_per_worker"],
        }

        cls = CustomEnvPolicyEvaluator.as_remote(**remote_args).remote

        return [
            self._make_evaluator(cls, env_creator, policy_graph, i + 1,
                                 self.config) for i in range(count)
        ]
       