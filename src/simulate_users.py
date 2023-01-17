# import functions
from maxent_irl import *
from toy_assembly import *

# import python libraries
import pickle
import numpy as np
from copy import deepcopy
import pandas as pd

# canonical_features = [[0.837, 0.244, 0.282],
#                       [0.212, 0.578, 0.018],
#                       [0.712, 0.911, 0.418],
#                       [0.462, 0.195, 0.882],
#                       [0.962, 0.528, 0.618],
#                       [0.056, 0.861, 0.218]]

complex_features = [[0.950, 0.033, 0.180],
                    [0.044, 0.367, 0.900],
                    [0.544, 0.700, 0.380],
                    [0.294, 0.145, 0.580],
                    [0.794, 0.478, 0.780],
                    [0.169, 0.811, 0.041],
                    [0.669, 0.256, 0.980],
                    [0.419, 0.589, 0.241],
                    [0.919, 0.922, 0.441],
                    [0.106, 0.095, 0.641]]

weights = [[0.60, 0.20, 0.20],
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.80, 0.10],
            [0.20, 0.20, 0.60],
            [0.10, 0.10, 0.80],
            [0.40, 0.40, 0.20],
            [0.40, 0.20, 0.40],
            [0.20, 0.40, 0.40],
            [0.40, 0.30, 0.30],
            [0.30, 0.40, 0.30],
            [0.50, 0.30, 0.20],
            [0.50, 0.20, 0.30],
            [0.30, 0.50, 0.20],
            [0.20, 0.50, 0.30],
            [0.30, 0.20, 0.50],
            [0.20, 0.30, 0.50]]

FILE_SUFFIX = "exp50_feat3_metric_dispersion_space_normal"
FILE_SUFFIX_TRANS = "exp50_trans_metric_dispersion_space_normal"

# Doesn't work past 6?
for task_class in ["best", "worst"]:
    with open('rirl/' + task_class + '_' + FILE_SUFFIX + ".pkl", "rb") as f:
        task_features = pickle.load(f)
    with open('rirl/' + task_class + '_' + FILE_SUFFIX_TRANS + ".pkl", "rb") as f:
        task_transitions = pickle.load(f)
    for action_space_size in sorted(list(task_features.keys())):
        print("Action space", action_space_size)
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
            X = ComplexTask(complex_features)
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
                canonical_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
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
