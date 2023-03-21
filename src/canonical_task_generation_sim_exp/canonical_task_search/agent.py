from abc import ABC
import numpy as np
import copy
from typing import List, Dict

from canonical_task_generation_sim_exp.canonical_task_search.task import RIRLTask
from canonical_task_generation_sim_exp.canonical_task_search.vi import value_iteration

class Agent(ABC):
    def __init__(self, task, max_reward=10, discount_factor=0.8):
        self.task = task
        self.action_space = self.task.actions
        self.num_actions = self.task.num_actions
        super(Agent)

    def act(self, state) -> int:
        pass

class GreedyAgent(Agent):
    """Defines a policy to execute on a specified task"""
    def __init__(self, task, max_reward=10, discount_factor=0.8):
        super().__init__(task,
            max_reward=max_reward,
            discount_factor=discount_factor)
        self.feat_weights = np.ones(self.task.num_features)
        self.state_rewards = GreedyAgent.rewards(self.task.states, self.task.features, self.feat_weights)

    def act(self, state) -> int:
        # State encodes previously executed actions
        # Get features for each executed task * feat weights for the agent
        # Sum to get total reward currently earned
        best_reward = 0
        best_action = None
        best_next_state = None

        shuffled_actions = copy.deepcopy(self.action_space)
        np.random.shuffle(shuffled_actions)

        for a in shuffled_actions:
            prob, next_state = self.task.transition(state, a)
            if next_state is not None:
                next_reward = self.state_rewards[RIRLTask.state_to_key(next_state)]
                if best_reward < next_reward:
                    best_action = a
                    best_reward = next_reward
                    best_next_state = next_state

        print(f"Agent: {state} -> {best_next_state} (action: {best_action}): Reward: {best_reward}")
        return best_action

    @staticmethod
    def rewards(states, features, weights) -> Dict[bytearray, float]:
        rewards = {}
        for s in states:
            rewards[RIRLTask.state_to_key(s)] = VIAgent.reward(s, features, weights)
        return rewards

    @staticmethod
    def reward(state, features, weights) -> float:
        return np.sum(features[np.where(state)] * weights)

class VIAgent(Agent):
    """Defines a policy to execute on a specified task"""
    def __init__(self, task, feat_weights: np.array, max_reward=10, discount_factor=0.8, verbose=False):
        super().__init__(task,
            max_reward=max_reward,
            discount_factor=discount_factor)
        self.feat_weights = feat_weights
        self.state_rewards = VIAgent.rewards(self.task.states, self.task.action_features, self.feat_weights)
        self.qf, self.vf, self.op_actions = value_iteration(self.task.states, RIRLTask.state_to_key, self.action_space, self.task.transition, self.state_rewards, self.task.terminal_states)
        # Count for each state how many actions have the same Q value ahead of time

        self.ambiguous_states = {}
        for s in self.task.states:
            unique_qs = set(self.qf[RIRLTask.state_to_key(s)])
            self.ambiguous_states[RIRLTask.state_to_key(s)] = len(self.action_space) - len(unique_qs)

        #TODO: Keep track of the accumulated features seen during traversial.
        # Expect a 1xnum_feat mat
        self.cumulative_seen_state_features = np.zeros_like(self.feat_weights)
        # THIS WOULD NORMALLY BE UNOBSERVABLE RIGHT?
        self.verbose = verbose

    def act(self, state) -> int:
        qs = self.qf[RIRLTask.state_to_key(state)]
        max_q = -np.inf
        best_action = None
        best_next_state = None

        for a, q in qs.items():
            #TODO: SHOULD WE SAMPLE FROM EQUALLY GOOD Q VALUES? DO THIS WITH THE FEATURE COUNT WORK
            if q > max_q:
                # NOTE: Only valid if transition prob is 1.0 (right now hard coded)
                _, next_state = self.task.transition(state, a)
                if next_state is not None:
                    best_action = a
                    best_next_state = next_state
                    max_q = q

        num_ties = 0
        all_options = []
        for a, q in qs.items():
            if q == max_q:
                num_ties += num_ties
                all_options.append(a)

        #if num_ties > 0:
        #    breakpoint()

        if self.verbose:
            print(f"Agent: {state} -> {best_next_state} (action: {best_action}, options: {all_options}): Q: {max_q}")

        # self.cumulative_seen_state_features += self.task.state_features[self.task.state_key_to_state_idx[RIRLTask.state_to_key(state)]]
        _, next_state = self.task.transition(state, best_action)
        self.cumulative_seen_state_features += self.task.state_features[self.task.state_key_to_state_idx[RIRLTask.state_to_key(next_state)]]

        return best_action, num_ties, all_options

    @staticmethod
    def rewards(states, features, weights) -> Dict[bytearray, float]:
        rewards = {}
        for s in states:
            rewards[RIRLTask.state_to_key(s)] = VIAgent.reward(s, features, weights)
        return rewards

    @staticmethod
    def reward(state, features, weights) -> float:
        return np.sum(features[np.where(state)] * weights)