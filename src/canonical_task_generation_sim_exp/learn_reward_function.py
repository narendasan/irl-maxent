import numpy as np
import pandas as pd
from typing import List, Tuple
from copy import deepcopy
from dask.distributed import Client, LocalCluster

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.vi import value_iteration_numba as value_iteration
from canonical_task_generation_sim_exp.lib.irl import maxent_irl, get_trajectories, boltzman_likelihood, predict_trajectory, online_predict_trajectory
from canonical_task_generation_sim_exp.irl_maxent import optimizer as O
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path

def learn_reward_func(canonical_task: CanonicalTask,
                      canonical_demo: List[int],
                      init: O,
                      weight_samples: np.array,
                      algorithm: str = "maxent",
                      n_train_samples = 50,
                      test_canonical: bool = False) -> Tuple[np.array, float]:

    print(canonical_task, canonical_demo, init, weight_samples)

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=2.0))

    canonical_demos = [list(canonical_demo)]
    canonical_trajectories = get_trajectories(canonical_task.states, canonical_demos, canonical_task.transition)

    # state features
    canonical_features = np.array([canonical_task.get_features(state) for state in canonical_task.states])
    #canonical_features /= np.linalg.norm(canonical_features, axis=0)
    canonical_features = np.nan_to_num(canonical_features)
    canonical_actions = list(range(len(canonical_features)))

    if algorithm == "maxent":
        #print("Training using Max-Entropy IRL ...")
        _, canonical_weights = maxent_irl(canonical_task, canonical_features, canonical_trajectories, optim, init)

    elif algorithm == "bayes":
        #print("Training using Bayesian IRL ...")
        posteriors = []
        weight_priors = np.ones(n_train_samples) / n_train_samples
        for n_sample in range(n_train_samples):
            sample = weight_samples[n_sample]
            all_canonical_trajectories = canonical_task.enumerate_trajectories([canonical_actions])
            likelihood_all_trajectories, _ = boltzman_likelihood(canonical_features, all_canonical_trajectories, sample)
            likelihood_user_demo, demo_reward = boltzman_likelihood(canonical_features, np.array(canonical_trajectories), sample)
            likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_trajectories)
            bayesian_update = (likelihood_user_demo[0] * weight_priors[n_sample])

            posteriors.append(bayesian_update)
        posteriors = list(posteriors / np.sum(posteriors))

        # select the MAP (maximum a posteriori) weight estimate
        max_posterior = max(posteriors)
        canonical_weights = weight_samples[posteriors.index(max_posterior)]

    else:
        canonical_weights = None

    if test_canonical:
        canonical_rewards = canonical_features.dot(canonical_weights)
        qf_abstract, _, _ = value_iteration(np.array(canonical_task.actions), np.array(canonical_task.trans_prob_mat), np.array(canonical_task.trans_state_mat), canonical_rewards, np.array(canonical_task.terminal_idx))
        predict_scores, predict_sequence_canonical, _ = predict_trajectory(qf_abstract, canonical_task.states, canonical_demos, canonical_task.transition)
        acc = np.mean(predict_scores, axis=0)

        return (canonical_weights, acc, predict_sequence_canonical)
    else:
        return (canonical_weights, None, None)

def load_learned_weights(kind: str, args) -> pd.DataFrame:
    p = out_path(args, kind="data", owner="learned_weights", load=True)
    weights_df = pd.read_csv(p / f"{kind}_learned_weights_archive.csv", index_col=[0,1,2,3,4], converters={"learned_weights":serialization.from_space_sep_list})
    weights_df["learned_weights"] = weights_df["learned_weights"].apply(np.array)
    return weights_df

def save_learned_weights(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="learned_weights")

    with (p / f"{kind}_learned_weights_archive.csv").open("w") as f:
        task_df.to_csv(f)

def train(
    dask_client: Client,
    canonical_task_archive: pd.DataFrame,
    complex_task_archive: pd.DataFrame,
    user_demos: pd.DataFrame,
    feat_size: int,
    canonical_action_space_size: int,
    complex_action_space_size: int,
    n_train_samples: int = 50
) -> pd.DataFrame:

    canonical_task_info = canonical_task_archive.loc[(feat_size, canonical_action_space_size)]
    canonical_task = CanonicalTask(canonical_task_info["features"], canonical_task_info["preconditions"])
    canonical_task.set_end_state(list(range(len(canonical_task_info["features"]))))
    canonical_task.enumerate_states()
    canonical_task.set_terminal_idx()

    complex_task_set = complex_task_archive.xs((feat_size, complex_action_space_size), level=["feat_dim", "num_actions"])
    user_demo_set = user_demos.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])


    train_args = []
    for task_id, demo_df in user_demo_set.groupby(level=["complex_task_id"]):
        complex_task_info = complex_task_set.iloc[task_id]
        complex_task = ComplexTask(complex_task_info["features"], complex_task_info["preconditions"])
        complex_task.set_end_state(list(range(len(complex_task_info["features"]))))
        complex_task.enumerate_states()
        complex_task.set_terminal_idx()

        for uid, demos in demo_df.groupby(level=["uid"]):
            canonical_demo = demos.loc[(task_id, uid)]["canonical_demo"]

            # select initial distribution of weights
            init = O.Constant(0.5)
            weight_samples = np.random.uniform(0., 1., (n_train_samples, feat_size))
            d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
            weight_samples = weight_samples / d

            train_args.append((task_id, uid, deepcopy(canonical_task), deepcopy(canonical_demo), deepcopy(init), deepcopy(weight_samples)))

    futures = dask_client.map(lambda t: learn_reward_func(t[2], t[3], t[4], t[5], test_canonical=True), train_args)
    training_results = dask_client.gather(futures)

    learned_weights_dict = {}
    for a, r in zip(train_args, training_results):
        print("=======================")
        print("Task:", a[0])
        print("User:", a[1])
        print("Weights have been learned for the canonical task! Hopefully.")
        print("Weights -", r[0])
        print("Canonical task:")
        print("     demonstration -", a[3])
        if r[2] is not None:
            print("     predicted demo -", r[2])

        if r[1] is not None:
            print("predict (abstract) -", r[1])

        learned_weights_dict[(feat_size, canonical_action_space_size, complex_action_space_size, a[0], a[1])] = (r[0], r[1])

    weights_idx = pd.MultiIndex.from_tuples(learned_weights_dict, names=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"])
    weights_set = list(learned_weights_dict.values())
    learned_weights_df = pd.DataFrame(weights_set, index=weights_idx, columns=["learned_weights", "canonical_task_acc"])

    return learned_weights_df

def main(args):

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)