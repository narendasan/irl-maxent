from dask.distributed import LocalCluster, Client
from typing import Tuple
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt

from canonical_task_generation_sim_exp.lib.arguments import parser, args_to_prefix, out_path
from canonical_task_generation_sim_exp.lib.hierarchal_task_networks import checkHTN
from canonical_task_generation_sim_exp.canonical_task_search import search
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS


def generate_complex_task(num_actions: int = 5, num_features: int = 3, feature_space = None):

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

def create_complex_task_archive(action_space_range: Tuple = (2, 10),
                                feat_space_range: Tuple = (3, 5),
                                num_tasks_per_quadrant: int = 10) -> pd.DataFrame:

    tasks = {}
    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        for a in range(action_space_range[0], action_space_range[1] + 1):
            for i in range(num_tasks_per_quadrant):
                result = generate_complex_task(a, f)
                tasks[(f, a, i)] = list(result)

    task_labels = list(tasks.keys())
    task_idx = pd.MultiIndex.from_tuples(task_labels, name=["feat_dim", "num_actions", ""])
    task_defs = list(tasks.values())
    task_df = pd.DataFrame(task_defs, index=task_idx, columns=["features", "preconditions"])

    return task_df

def main(args):
    task_df = create_complex_task_archive(action_space_range=(2, args.max_action_space_size),
                                         feat_space_range=(3, args.max_feature_space_size))

    p = out_path(args, kind="data", owner="complex_task_archive")

    with (p / "task_archive.csv").open("w") as f:
        task_df.to_csv(f)

    print(task_df)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)