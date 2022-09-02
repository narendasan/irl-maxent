from abc import ABC
import numpy as np
import copy
from vi import value_iteration
from typing import List, Dict

class Agent(ABC):
    def __init__(self, task, max_reward=10, discount_factor=0.8):
        self.task = task
        self.action_space = self.task.actions
        self.num_actions = self.task.num_actions
        super(Agent)

    def act(self, state) -> int:
        pass

def to_key(x: np.array):
    assert(x.dtype == np.uint8)
    return hash(x.data.tobytes())

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
                next_reward = self.state_rewards[to_key(next_state)]
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
            rewards[to_key(s)] = VIAgent.reward(s, features, weights)
        return rewards

    @staticmethod
    def reward(state, features, weights) -> float:
        return np.sum(features[np.where(state)] * weights)

class VIAgent(Agent):
    """Defines a policy to execute on a specified task"""
    def __init__(self, task, max_reward=10, discount_factor=0.8):
        super().__init__(task,
            max_reward=max_reward,
            discount_factor=discount_factor)
        self.feat_weights = np.ones(self.task.num_features)
        self.state_rewards = VIAgent.rewards(self.task.states, self.task.features, self.feat_weights)
        self.qf, self.vf, self.op_actions = value_iteration(self.task.states, to_key, self.action_space, self.task.transition, self.state_rewards, self.task.terminal_states)

    def act(self, state) -> int:
        qs = self.qf[to_key(state)]
        max_q = 0
        best_action = None
        best_next_state = None

        for a, q in qs.items():
            if q > max_q:
                _, next_state = self.task.transition(state, a)
                if next_state is not None:
                    best_action = a
                    best_next_state = next_state
                    max_q = q

        print(f"Agent: {state} -> {best_next_state} (action: {best_action}): Q: {max_q}")
        return best_action

    @staticmethod
    def rewards(states, features, weights) -> Dict[bytearray, float]:
        rewards = {}
        for s in states:
            rewards[to_key(s)] = VIAgent.reward(s, features, weights)
        return rewards

    @staticmethod
    def reward(state, features, weights) -> float:
        return np.sum(features[np.where(state)] * weights)