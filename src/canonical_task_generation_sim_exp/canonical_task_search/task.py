from copy import deepcopy
import numpy as np
import math
from dataclasses import dataclass

class RIRLTask:
    def __init__(self, features, preconditions):
        self.num_actions, self.num_features = np.shape(features)
        self.actions = self._generate_action_space()
        self.action_features = np.array(features)
        self.action_preconditions = np.array(preconditions)
        self.states = self._generate_state_space()
        assert(np.equal(self.states[len(self.states) - 1], np.ones((self.num_actions))).all())
        self.state_features = self._generate_state_features_from_action_features()
        assert(self.state_features.shape == (self.states.shape[0], self.num_features))
        self.terminal_states = [self.states[len(self.states) - 1]]
        self.state_key_to_state_idx = {RIRLTask.state_to_key(s) : i for i, s in enumerate(self.states)}

    @staticmethod
    def state_to_key(x: np.array):
        assert(x.dtype == np.uint8)
        return hash(x.data.tobytes())

    def _generate_state_space(self):
        max_val = (2 ** self.num_actions)
        state_list = [np.array(list(map(int, np.binary_repr(i, width=self.num_actions)))) for i in range(max_val)]
        states = np.vstack(state_list)
        assert states.shape == (max_val, self.num_actions)

        return states.astype(np.uint8)

    def _generate_action_space(self):
        return np.array(range(self.num_actions))

    def _generate_state_features_from_action_features(self):
        return np.array([np.sum(self.action_features[np.where(taken_actions_at_state)], axis=0) for taken_actions_at_state in self.states])

    def r_max(self):
        # THIS IS R_MAX AS LONG AS FEATURE WEIGHTS ARE NO GREATER THAN 1
        return np.sum(self.features)

    # @staticmethod
    def transition(self, s_from, a):

        satisfy_preconditions = [s_from[ap_idx] for ap_idx, ap in enumerate(self.action_preconditions[a]) if ap]

        # Action has been performed already
        if s_from[a] == 0 and all(satisfy_preconditions):
            s_to = deepcopy(s_from)
            s_to[a] = 1
            return 1.0, s_to.astype(np.uint8)
        else:
            return 0.0, None


    def __str__(self) -> str:
        return f"""
Task:
    Action Space: {self.num_actions}
    Features: \n{self.action_features}
"""

    def __repr__(self):
        return self.__str__()

def recursive_dependencies(actions, dependencies, result=None):
    """
    Recursively populate all dependencies of actions.
    """

    if result is None:
        result = set()
    for a in actions:
        result.update(dependencies[a])
        recursive_dependencies(dependencies[a], dependencies, result)

    return result

def checkHTN(precondition, all_preconditions):
    """
    Function to check if new action affects the preconditions of other actions.
    Args:
        precondition: preconditions of new action
        all_preconditions: preconditions of all actions in the task

    Returns:
        True if new action can be added to the task without affecting other actions.
    """

    # dependency for each action
    dependencies = [set(np.where(oa_dependency)[0]) for oa_dependency in all_preconditions]

    # dependency for new action
    a_dependency = set(np.where(precondition)[0])
    num_dependencies = len(a_dependency)

    if num_dependencies > 0:

        verification = True

        # check if new action affects the dependency of previous actions
        for oa, oa_dependency in enumerate(dependencies):
            if oa not in a_dependency:
                common_dependencies = a_dependency.intersection(oa_dependency)
                unique_dependencies = a_dependency.symmetric_difference(oa_dependency)
                if common_dependencies and unique_dependencies:
                    # check if the common dependencies are dependent on the unique
                    all_oa_dependencies = recursive_dependencies(common_dependencies, dependencies)
                    if all_oa_dependencies != unique_dependencies:
                        verification = False
                        break
    else:
        # no check required for actions without any dependencies
        verification = True

    return verification


def generate_task(num_actions, num_features, feature_space=None):

    # TODO: directly take feature space as input
    if not feature_space:
        if num_actions < 3:
            feature_bounds = [(0, num_actions),  # which part
                              (0, num_actions)]  # which tool
        else:
            feature_bounds = [(0, math.ceil(num_actions/2)),
                              (0, math.ceil(num_actions/2))]

        feature_space = []
        for lb, ub in feature_bounds:
            feature_space.append([f_val for f_val in range(lb, ub, 1)])

    task_actions, task_preconditions = [], []
    for i in range(num_actions):
        new_action = []
        for j in range(num_features):
            if j < len(feature_space):
                new_action.append(np.random.choice(feature_space[j]))
            else:
                new_action.append(np.random.random())

        precondition_verified = False
        while not precondition_verified:
            action_precondition = np.zeros(num_actions)
            for oa in range(len(task_actions)):
                dep = np.random.choice([0, 1], p=[0.60, 0.40])
                if dep == 1:
                    action_precondition[oa] = 1

            precondition_verified = checkHTN(action_precondition, task_preconditions)

        task_actions.append(new_action)
        task_preconditions.append(action_precondition)

    return task_actions, task_preconditions

if __name__ == "__main__":
    task = RIRLTask(features=np.random.random((5, 3)))
    print(task.actions, task.states)