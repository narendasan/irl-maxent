import math
import numpy as np
from typing import Tuple

from canonical_task_generation_sim_exp.lib.hierarchal_task_networks import checkHTN


def generate_task(num_actions, num_features, feature_space=None, precondition_probs: Tuple[float, float] = (0.4, 0.6)):

    # TODO: directly take feature space as input
    if feature_space is not None:
        if num_actions < 3:
            feature_bounds = [(0, num_actions),  # which part
                              (0, num_actions)]  # which tool
        else:
            feature_bounds = [(0, math.ceil(num_actions/2)),
                              (0, math.ceil(num_actions/2))]

        feature_space = []
        # for lb, ub in feature_bounds:
        #     feature_space.append([f_val for f_val in range(lb, ub, 1)])

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
                dep = np.random.choice([0, 1], p=precondition_probs)
                if dep == 1:
                    action_precondition[oa] = 1

            precondition_verified = checkHTN(action_precondition, task_preconditions)

        task_actions.append(new_action)
        task_preconditions.append(action_precondition)

    return task_actions, task_preconditions