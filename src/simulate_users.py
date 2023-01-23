# import functions
from maxent_irl import *
from toy_assembly import *

import scipy.stats
# import python libraries
import pickle
import numpy as np
from copy import deepcopy
import pandas as pd
import math

from rirl import generate_task

# canonical_features = [[0.837, 0.244, 0.282],
#                       [0.212, 0.578, 0.018],
#                       [0.712, 0.911, 0.418],
#                       [0.462, 0.195, 0.882],
#                       [0.962, 0.528, 0.618],
#                       [0.056, 0.861, 0.218]]

# complex_features = [[0.950, 0.033, 0.180],
#                     [0.044, 0.367, 0.900],
#                     [0.544, 0.700, 0.380],
#                     [0.294, 0.145, 0.580],
#                     [0.794, 0.478, 0.780],
#                     [0.169, 0.811, 0.041],
#                     [0.669, 0.256, 0.980],
#                     [0.419, 0.589, 0.241],
#                     [0.919, 0.922, 0.441],
#                     [0.106, 0.095, 0.641]]

# weights = [[0.60, 0.20, 0.20],
#             [0.80, 0.10, 0.10],
#             [0.20, 0.60, 0.20],
#             [0.10, 0.80, 0.10],
#             [0.20, 0.20, 0.60],
#             [0.10, 0.10, 0.80],
#             [0.40, 0.40, 0.20],
#             [0.40, 0.20, 0.40],
#             [0.20, 0.40, 0.40],
#             [0.40, 0.30, 0.30],
#             [0.30, 0.40, 0.30],
#             [0.50, 0.30, 0.20],
#             [0.50, 0.20, 0.30],
#             [0.30, 0.50, 0.20],
#             [0.20, 0.50, 0.30],
#             [0.30, 0.20, 0.50],
#             [0.20, 0.30, 0.50]]

import argparse
parser = argparse.ArgumentParser(description='Repeated IRL experiements')
parser.add_argument('--num-experiments', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--weight-samples', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--max-action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions')
parser.add_argument('--feature-space-size', type=int,
                    default=3,
                    help='Dimensionality of feature space describing actions')
parser.add_argument('--max-experiment-len', type=int,
                    default=100,
                    help='Maximum number of steps taken in each experiment')
parser.add_argument("--verbose", action='store_true', help='Print selected tasks')
parser.add_argument("--load-from-file", type=str, help="Load a task from a saved file")
parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes to run experiements")
parser.add_argument("--metric", type=str, default="unique-trajectories",
                    help="What metric to use to determine if a task is good a distingushing reward functions")
parser.add_argument("--weight-space", type=str, default="normal", help="What space to sample weights from")
parser.add_argument("--iteration", type=int, default=0, help="What iteration of the experiment")
parser.add_argument("--headless", action='store_true', help='Dont show figures')

args = parser.parse_args()

FILE_SUFFIX = f"exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}"
FILE_SUFFIX_TRANS = f"exp{args.num_experiments}_trans_metric_{args.metric}_space_{args.weight_space}_{args.iteration}"

#complex_features = sample_complex_features(feat_space_size=args.feature_space_size)

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


def sample_users(feat_space_size):
    shape = (2, feat_space_size)
    rng = scipy.stats.qmc.Halton(d=shape[1], scramble=False)
    return rng.random(n=shape[0] + 1)[1:]  # Skip the first one which is always 0,0,0 when scramble is off

weights = sample_users(feat_space_size=args.feature_space_size)

with open(f"data/user_weights_{FILE_SUFFIX}.pkl", "wb") as f:
    pickle.dump(weights, f)

# Doesn't work past 6?
for task_class in ["best", "random", "worst"]:
    with open("results/" + task_class + '_' + FILE_SUFFIX + ".pkl", "rb") as f:
        task_features = pickle.load(f)
    with open("results/" + task_class + '_' + FILE_SUFFIX_TRANS + ".pkl", "rb") as f:
        task_transitions = pickle.load(f)
    for action_space_size in sorted(list(task_features.keys())):
        print("Action space", action_space_size)
        complex_features, complex_transitions = generate_task(15, args.feature_space_size)

        with open(f"data/complex_task_features_action_space_size_{2 * action_space_size}_{FILE_SUFFIX}.pkl", "wb") as f:
            pickle.dump(complex_features, f)

        with open(f"data/complex_task_transitions_action_space_size_{2 * action_space_size}_{FILE_SUFFIX}.pkl", "wb") as f:
            pickle.dump(complex_transitions, f)

        for j in range(len(task_features[action_space_size])):
            canonical_features = task_features[action_space_size][j]
            canonical_transitions = task_transitions[action_space_size][j]

            # adversarial_weights = np.array(weights)
            # adversarial_weights[:, [0, 1, 2]] = adversarial_weights[:, [2, 0, 1]]

            canonical_actions = list(range(len(canonical_features)))
            complex_actions = list(range(len(complex_features)))

            # initialize canonical task
            C = CanonicalTask(canonical_features, canonical_transitions)
            C.set_end_state(canonical_actions)
            C.enumerate_states()
            C.set_terminal_idx()
            # all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

            # initialize actual task
            X = ComplexTask(complex_features, complex_transitions)
            X.set_end_state(complex_actions)
            X.enumerate_states()
            X.set_terminal_idx()
            # all_complex_trajectories = X.enumerate_trajectories([complex_actions])

            # loop over all users
            canonical_demos, complex_demos = [], []
            for i in range(len(weights)):

                print("=======================")
                print("User:", i)

                # using abstract features
                abstract_features = np.array([C.get_features(state) for state in C.states])
                canonical_abstract_features = abstract_features / (np.linalg.norm(abstract_features, axis=0) + 1e-10)
                canonical_abstract_features = np.nan_to_num(canonical_abstract_features)

                complex_abstract_features = np.array([X.get_features(state) for state in X.states])
                complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

                canonical_rewards = canonical_abstract_features.dot(weights[i])
                complex_rewards = complex_abstract_features.dot(weights[i])

                qf_canonical, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
                qf_complex, _, _ = value_iteration(X.states, X.actions, X.transition, complex_rewards, X.terminal_idx)

                canonical_demo = rollout_trajectory(qf_canonical, C.states, C.transition, canonical_actions)
                complex_demo = rollout_trajectory(qf_complex, X.states, X.transition, complex_actions)

                canonical_demos.append(canonical_demo)
                complex_demos.append(complex_demo)
                print("Canonical demo:", canonical_demo)
                print("  Complex demo:", complex_demo)

            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", weights)
            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_demos_{j}.csv", canonical_demos)
            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_demos_{j}.csv", complex_demos)
            # pickle.dump(all_canonical_trajectories, open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_trajectories_{j}.csv", "wb"))
            # pickle.dump(all_complex_trajectories, open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_trajectories_{j}.csv", "wb"))

print("Done.")
