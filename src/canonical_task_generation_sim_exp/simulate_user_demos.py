import scipy.stats
# import python libraries
import pickle
import numpy as np
from copy import deepcopy
import pandas as pd
import math
from typing import Tuple, List

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask
from canonical_task_generation_sim_exp.lib.vi import value_iteration
from canonical_task_generation_sim_exp.lib.irl import rollout_trajectory
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path

def simulate_user(user_weights: np.array, canonical_task: CanonicalTask, complex_task: ComplexTask) -> Tuple[List[int], List[int]]:
    # using abstract features
    abstract_features = np.array([canonical_task.get_features(state) for state in canonical_task.states])
    canonical_abstract_features = abstract_features / (np.linalg.norm(abstract_features, axis=0) + 1e-10)
    canonical_abstract_features = np.nan_to_num(canonical_abstract_features)

    complex_abstract_features = np.array([complex_task.get_features(state) for state in complex_task.states])
    complex_abstract_features /= (np.linalg.norm(complex_abstract_features, axis=0) + 1e-10)

    canonical_rewards = canonical_abstract_features.dot(user_weights)
    complex_rewards = complex_abstract_features.dot(user_weights)


    qf_canonical, _, _ = value_iteration(canonical_task.states, canonical_task.actions, canonical_task.transition, canonical_rewards, canonical_task.terminal_idx)
    qf_complex, _, _ = value_iteration(complex_task.states, complex_task.actions, complex_task.transition, complex_rewards, complex_task.terminal_idx)

    canonical_demo = rollout_trajectory(qf_canonical, canonical_task.states, canonical_task.transition, list(canonical_task.actions))
    complex_demo = rollout_trajectory(qf_complex, complex_task.states, complex_task.transition, list(complex_task.actions))

    print("Canonical demo:", canonical_demo)
    print("  Complex demo:", complex_demo)

    return (canonical_demo, complex_demo)

def load_demos(kind: str, args) -> pd.DataFrame:
    p = out_path(args, kind="data", owner="sim_user_demos")
    demo_df = pd.read_csv(p / f"{kind}_demo_archive.csv", index_col=[0,1,2,3,4], converters={"canonical_demo":serialization.from_list, "complex_demo": serialization.from_list})
    return demo_df

def save_demos(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="sim_user_demos")

    with (p / f"{kind}_demo_archive.csv").open("w") as f:
        task_df.to_csv(f)


def sim_demos(canonical_task_archive: pd.DataFrame,
              complex_task_archive: pd.DataFrame,
              users: np.array,
              feat_size: int,
              canonical_action_space_size: int,
              complex_action_space_size: int) -> pd.DataFrame:

    canonical_task_info = canonical_task_archive.loc[(feat_size, canonical_action_space_size)]
    canonical_task = CanonicalTask(canonical_task_info["features"], canonical_task_info["preconditions"])
    canonical_task.set_end_state(list(range(len(canonical_task_info["features"]))))
    canonical_task.enumerate_states()
    canonical_task.set_terminal_idx()

    complex_task_set = complex_task_archive.xs((feat_size, complex_action_space_size), level=["feat_dim", "num_actions"])

    demo_dict = {}
    for i in range(len(complex_task_set)):
        print("+++++++++++++++++++++++++++++")
        print("Task:", i)
        complex_task_info = complex_task_set.iloc[i]
        complex_task = ComplexTask(complex_task_info["features"], complex_task_info["preconditions"])
        complex_task.set_end_state(list(range(len(complex_task_info["features"]))))
        complex_task.enumerate_states()
        complex_task.set_terminal_idx()

        for uid, u in enumerate(users):
            print("=======================")
            print("User:", uid)
            demo_dict[(feat_size, canonical_action_space_size, complex_action_space_size, i, uid)] = simulate_user(u, canonical_task, complex_task)


    demo_labels = list(demo_dict.keys())
    demos_idx = pd.MultiIndex.from_tuples(demo_labels, names=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"])
    demos = [[d[0], d[1]]for d in demo_dict.values()]

    demo_df = pd.DataFrame(demos, index=demos_idx, columns=["canonical_demo", "complex_demo"])

    return demo_df