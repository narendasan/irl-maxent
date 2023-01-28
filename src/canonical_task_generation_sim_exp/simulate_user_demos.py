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

from canonical_task_generation_sim_exp.simulated_task.assembly_task import AssemblyTask

def simulate_user(user, canonical_task: AssemblyTask, complex_task: AssemblyTask) -> Demo:


# Doesn't work past 6?
for task_class in ["best", "random", "worst"]:
    with open("results/" + task_class + '_' + FILE_SUFFIX + ".pkl", "rb") as f:
        task_features = pickle.load(f)
    with open("results/" + task_class + '_' + FILE_SUFFIX_TRANS + ".pkl", "rb") as f:
        task_transitions = pickle.load(f)
    for action_space_size in sorted(list(task_features.keys())):
        print("Action space", action_space_size)
        complex_features, complex_transitions = generate_task(15, args.feature_space_size)

        with open(f"data/{task_class}_complex_task_features_action_space_size_{2 * action_space_size}_{FILE_SUFFIX}.pkl", "wb") as f:
            np.save(f, complex_features, allow_pickle=True)

        with open(f"data/complex_task_transitions_action_space_size_{2 * action_space_size}_{FILE_SUFFIX}.pkl", "wb") as f:
            np.save(f, complex_transitions, allow_pickle=True)

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

                complex_trajectories = get_trajectories(X.states, complex_demos, X.transition)

            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", weights)
            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_demos_{j}.csv", canonical_demos)
            np.savetxt(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_demos_{j}.csv", complex_demos)
            # pickle.dump(all_canonical_trajectories, open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_trajectories_{j}.csv", "wb"))
            # pickle.dump(all_complex_trajectories, open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_trajectories_{j}.csv", "wb"))

