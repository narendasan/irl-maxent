from dask.distributed import LocalCluster, Client
from typing import Tuple, Any
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt

from canonical_task_generation_sim_exp.lib.action_space_range import complex_action_space_range
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.hierarchal_task_networks import checkHTN
from canonical_task_generation_sim_exp.lib.generate_tasks import generate_task
from canonical_task_generation_sim_exp.canonical_task_search import search



def generate_complex_task(num_actions: int = 5, num_features: int = 3):
    feat_space = np.array([0.1, 0.5, 0.9])
    feat_choices = np.repeat(feat_space[np.newaxis, ...], num_actions, axis=0)
    return generate_task(num_actions, num_features, feature_space=feat_choices, precondition_probs=(0.7, 0.3))

def create_complex_task_archive(action_space_range: Tuple = (10, 30),
                                feat_space_range: Tuple = (3, 5),
                                num_tasks_per_quadrant: int = 10) -> pd.DataFrame:

    tasks = {}
    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        for a in complex_action_space_range(action_space_range[0], action_space_range[1]):
            for i in range(num_tasks_per_quadrant):
                result = generate_complex_task(a, f)
                tasks[(f, a, i)] = list(result)

    task_labels = list(tasks.keys())
    task_idx = pd.MultiIndex.from_tuples(task_labels, name=["feat_dim", "num_actions", ""])
    task_defs = list(tasks.values())
    task_df = pd.DataFrame(task_defs, index=task_idx, columns=["features", "preconditions"])

    return task_df

def find_n_hardest_complex_tasks(dask_client: Client,
                                user_archive: pd.DataFrame,
                                action_space_range: Tuple = (10, 30),
                                feat_space_range: Tuple = (3, 5),
                                num_tasks_per_quadrant: int = 10,
                                metric: str = "dispersion",
                                num_sampled_tasks: int = 10,
                                num_sampled_agents: int = 10,
                                max_experiment_len: int = 100,
                                args: Any = None) -> pd.DataFrame:

    found_tasks = {}

    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        feat_user_df = user_archive.loc[[f]]
        feat_users = feat_user_df["users"]
        for a in complex_action_space_range(action_space_range[0], action_space_range[1]):
            result = search.find_n_best_tasks(dask_client=dask_client,
                                            agent_archive=feat_users,
                                            action_space_size=a,
                                            feat_space_size=f,
                                            metric=metric,
                                            num_sampled_tasks=num_sampled_tasks,
                                            num_sampled_agents=num_sampled_agents,
                                            max_experiment_len=max_experiment_len,
                                            num_tasks=num_tasks_per_quadrant,
                                            args=args)

            for i, task in enumerate(result.tasks):
                found_tasks[(f, a, i)] = task


    task_labels = list(found_tasks.keys())
    task_idx = pd.MultiIndex.from_tuples(task_labels, names=["feat_dim", "num_actions", ""])
    tasks = [[t.features, t.preconditions] for t in found_tasks.values()]

    task_df = pd.DataFrame(tasks, index=task_idx, columns=["features", "preconditions"])

    return task_df

def load_tasks(args) -> pd.DataFrame:
    p = out_path(args, kind="data", owner="complex_task_archive", load=True)
    task_df = pd.read_csv(p / "task_archive.csv", index_col=[0,1,2], converters={'features': serialization.from_list, 'preconditions': serialization.from_np_array})
    task_df["features"] = task_df["features"].apply(np.array)
    task_df["preconditions"] = task_df["preconditions"].apply(np.array)
    return task_df

def save_tasks(task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="complex_task_archive")

    with (p / "task_archive.csv").open("w") as f:
        task_df.to_csv(f)

def main(args):
    task_df = create_complex_task_archive(action_space_range=(2, args.max_complex_action_space_size),
                                         feat_space_range=(3, args.max_feature_space_size))


    save_tasks(task_df, args)
    print(task_df)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)