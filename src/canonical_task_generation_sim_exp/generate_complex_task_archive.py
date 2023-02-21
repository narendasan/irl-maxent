from dask.distributed import LocalCluster, Client
from typing import Tuple
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt

from canonical_task_generation_sim_exp.lib.action_space_range import complex_action_space_range
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.lib.hierarchal_task_networks import checkHTN
from canonical_task_generation_sim_exp.lib.generate_tasks import generate_task


def generate_complex_task(num_actions: int = 5, num_features: int = 3, feature_space = None):
    return generate_task(num_actions, num_features, feature_space, constraint_probs=(0.8, 0.2))

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