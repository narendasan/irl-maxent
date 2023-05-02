import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from dask.distributed import Client, LocalCluster


from typing import List, Tuple

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import ComplexTask
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS
from canonical_task_generation_sim_exp.lib.vi import value_iteration_numba as value_iteration
from canonical_task_generation_sim_exp.lib.irl import predict_trajectory, online_predict_trajectory
from canonical_task_generation_sim_exp.irl_maxent import optimizer as O

def evaluate_rf_acc(complex_task: ComplexTask,
        learned_weights: np.array,
        complex_user_demo: List[int],
        weight_samples: np.array = None,
        n_test_samples: int = None,
        posteriors: np.array = None,
        map_estimate: bool = True,
        init: O = None,
        algorithm: str = "maxent") -> Tuple[List[int], float]:

    complex_features = np.array([complex_task.get_features(state) for state in complex_task.states])
    complex_features /= np.linalg.norm(complex_features, axis=0) + 1e-10

    if map_estimate:
        transferred_weights = [learned_weights]
    elif algorithm == "bayes":
        weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples, p=posteriors)
        transferred_weights = weight_samples[weight_idx]
    else:
        transferred_weights = []

    for transferred_weight in transferred_weights:
        # transfer rewards to complex task
        complex_rewards = complex_features.dot(transferred_weight)

        # compute policy for transferred rewards
        qf_transfer, _, _ = value_iteration(np.array(complex_task.actions), np.array(complex_task.trans_prob_mat), np.array(complex_task.trans_state_mat), complex_rewards, np.array(complex_task.terminal_idx))

        # score for predicting user action at each time step
        if "online" in algorithm:
            if algorithm == "online_bayes":
                print("Online Prediction using Bayesian IRL ...")
                all_complex_trajectories = complex_task.enumerate_trajectories([complex_task.actions])
                p_score, predict_sequence, _ = online_predict_trajectory(complex_task, complex_user_demo,
                                                                    all_complex_trajectories,
                                                                    transferred_weight,
                                                                    complex_features,
                                                                    weight_samples, priors=[],
                                                                    sensitivity=0.0,
                                                                    consider_options=False,
                                                                    run_bayes=True)
            elif algorithm == "online_maxent":
                print("Online Prediction using Max Entropy IRL ...")
                optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))
                ol_optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))
                p_score, predict_sequence, _ = online_predict_trajectory(complex_task, complex_user_demo,
                                                                    all_complex_trajectories,
                                                                    transferred_weight,
                                                                    complex_features,
                                                                    weight_samples,
                                                                    sensitivity=0.0,
                                                                    consider_options=False,
                                                                    run_maxent=True,
                                                                    optim=ol_optim,
                                                                    init=init)
        else:
            p_score, predict_sequence, _ = predict_trajectory(qf_transfer, complex_task.states,
                                                            [complex_user_demo],
                                                            complex_task.transition,
                                                            sensitivity=0.0,
                                                            consider_options=False)
            predict_sequence = [item for sublist in predict_sequence for item in sublist]

    predict_score = np.mean(p_score, axis=0)

    return predict_sequence, predict_score


def avg_complex_task_acc(task_df: pd.DataFrame) -> pd.DataFrame:
    task_acc = task_df.groupby(level=["feat_dim","num_canonical_actions","num_complex_actions", "complex_task_id"]).mean()
    complex_as_acc = task_acc.groupby(level=["feat_dim","num_canonical_actions","num_complex_actions"]).mean()

    return complex_as_acc

def avg_task_acc(task_df: pd.DataFrame) -> pd.DataFrame:
    task_acc = task_df.groupby(level=["feat_dim","num_canonical_actions","num_complex_actions", "complex_task_id"]).mean()
    acc = task_acc.groupby(level=["feat_dim","num_canonical_actions","num_complex_actions"]).mean()
    return acc

def load_processed_results(kind: str, args) -> pd.DataFrame:
    p = out_path(args, kind="results", owner="avg_rf_acc")
    task_df = pd.read_csv(p / f"{kind}_avg_rf_acc.csv", index_col=[0,1,2])
    return task_df

def load_eval_results(kind: str, args) -> pd.DataFrame:
    p = out_path(args, kind="results", owner="learned_rf_acc")
    task_df = pd.read_csv(p / f"{kind}_learned_rf_acc.csv", index_col=[0,1,2,3,4, 5], converters={"predicted_complex_demo":serialization.from_list})
    return task_df

def save_processed_results(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="results", owner="avg_rf_acc")

    with (p / f"{kind}_avg_rf_acc.csv").open("w") as f:
        task_df.to_csv(f)

def save_eval_results(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="results", owner="learned_rf_acc")

    with (p / f"{kind}_learned_rf_acc.csv").open("w") as f:
        task_df.to_csv(f)

def eval(
    dask_client: Client,
    complex_task_archive: pd.DataFrame,
    learned_rf_weights: pd.DataFrame,
    user_complex_demos: pd.DataFrame,
    feat_size: int,
    canonical_action_space_size: int,
    complex_action_space_size: int
) -> pd.DataFrame:

    complex_task_set = complex_task_archive.xs((feat_size, complex_action_space_size), level=["feat_dim", "num_actions"])
    user_demo_set = user_complex_demos.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])
    learned_weights_set = learned_rf_weights.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])

    eval_args = []
    for ((task_id, task_demo_df),(_, task_learned_weights)) in zip(user_demo_set.groupby(level=["complex_task_id"]), learned_weights_set.groupby(level=["complex_task_id"])):
        complex_task_info = complex_task_set.iloc[task_id]
        complex_task = ComplexTask(complex_task_info["features"], complex_task_info["preconditions"])
        complex_task.set_end_state(list(range(len(complex_task_info["features"]))))
        complex_task.enumerate_states()
        complex_task.set_terminal_idx()

        for ((uid, user_task_demos),(_, user_task_learned_weights)) in zip(task_demo_df.groupby(level=["uid"]), task_learned_weights.groupby(level=["uid"])):
            complex_demo = user_task_demos.loc[(task_id, uid)]["complex_demo"]
            weights = user_task_learned_weights.loc[(task_id, uid)]["learned_weights"]
            eval_args.append((task_id, uid, deepcopy(complex_task), deepcopy(weights), deepcopy(complex_demo)))

    futures = dask_client.map(lambda e: evaluate_rf_acc(e[2], e[3], e[4]), eval_args)
    eval_results = dask_client.gather(futures)

    rf_acc = {}
    for a, r in zip(eval_args, eval_results):
        print("=======================")
        print("Task:", a[0])
        print("User:", a[1])
        print(f" Avg: {r[1]}")
        print("\n")
        print("Complex task:")
        print("   demonstration -", a[-1])
        print("     predictions -", r[0])
        rf_acc[(feat_size, canonical_action_space_size, complex_action_space_size, a[0], a[1])] = r





    rf_acc_labels = list(rf_acc.keys())
    rf_acc_idx = pd.MultiIndex.from_tuples(rf_acc_labels, names=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"])
    rf_acc_set = list(rf_acc.values())
    rf_acc_df = pd.DataFrame(rf_acc_set, index=rf_acc_idx, columns=["predicted_complex_demo", "complex_task_acc"])

    return rf_acc_df

def main(args):

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)