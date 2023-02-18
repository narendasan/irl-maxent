import numpy as np
import pandas as pd
from typing import List, Tuple

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask
from canonical_task_generation_sim_exp.lib.vi import value_iteration
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

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

    canonical_demos = [list(canonical_demo)]
    canonical_trajectories = get_trajectories(canonical_task.states, canonical_demos, canonical_task.transition)

    # state features
    canonical_features = np.array([canonical_task.get_features(state) for state in canonical_task.states])
    canonical_features /= np.linalg.norm(canonical_features, axis=0)
    canonical_features = np.nan_to_num(canonical_features)
    canonical_actions = list(range(len(canonical_features)))

    if algorithm == "maxent":
        print("Training using Max-Entropy IRL ...")
        _, canonical_weights = maxent_irl(canonical_task, canonical_features, canonical_trajectories, optim, init)

    elif algorithm == "bayes":
        print("Training using Bayesian IRL ...")
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

    print("Weights have been learned for the canonical task! Hopefully.")
    print("Weights -", canonical_weights)

    acc = None

    if test_canonical:
        canonical_rewards = canonical_features.dot(canonical_weights)
        qf_abstract, _, _ = value_iteration(canonical_task.states, canonical_task.actions, canonical_task.transition, canonical_rewards, canonical_task.terminal_idx)
        predict_scores, predict_sequence_canonical, _ = predict_trajectory(qf_abstract, canonical_task.states, canonical_demos, canonical_task.transition)

        print("Canonical task:")
        print("     demonstration -", canonical_demo)
        print("     predicted demo -", predict_sequence_canonical)
        print("predict (abstract) -", predict_scores)
        acc = np.mean(predict_scores, axis=0)

    return (canonical_weights, acc)

def load_learned_weights(kind: str, args) -> pd.DataFrame:
    p = out_path(args, kind="data", owner="learned_weights", load=True)
    task_df = pd.read_csv(p / f"{kind}_learned_weights_archive.csv", index_col=[0,1,2,3,4])
    return task_df

def save_learned_weights(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="learned_weights")

    with (p / f"{kind}_learned_weights_archive.csv").open("w") as f:
        task_df.to_csv(f)

def train(canonical_task_archive: pd.DataFrame,
        complex_task_archive: pd.DataFrame,
        users: np.array,
        user_demos: pd.DataFrame,
        feat_size: int,
        canonical_action_space_size: int,
        complex_action_space_size: int,
        n_train_samples: int = 50) -> pd.DataFrame:

    canonical_task_info = canonical_task_archive.loc[(feat_size, canonical_action_space_size)]
    canonical_task = CanonicalTask(canonical_task_info["features"], canonical_task_info["preconditions"])
    canonical_task.set_end_state(list(range(len(canonical_task_info["features"]))))
    canonical_task.enumerate_states()
    canonical_task.set_terminal_idx()

    complex_task_set = complex_task_archive.xs((feat_size, complex_action_space_size), level=["feat_dim", "num_actions"])
    user_demo_set = user_demos.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])

    learned_weights_dict = {}
    for task_id, demo_df in user_demo_set.groupby(level=["complex_task_id"]):
        print("+++++++++++++++++++++++++++++")
        print("Task:", task_id)
        complex_task_info = complex_task_set.iloc[task_id]
        complex_task = ComplexTask(complex_task_info["features"], complex_task_info["preconditions"])
        complex_task.set_end_state(list(range(len(complex_task_info["features"]))))
        complex_task.enumerate_states()
        complex_task.set_terminal_idx()

        for uid, demos in demo_df.groupby(level=["uid"]):
            print("=======================")
            print("User:", uid)
            canonical_demo = demos.loc[(task_id, uid)]["canonical_demo"]

            # select initial distribution of weights
            init = O.Constant(0.5)
            weight_samples = np.random.uniform(0., 1., (n_train_samples, feat_size))
            d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
            weight_samples = weight_samples / d

            weights, acc = learn_reward_func(canonical_task, canonical_demo, init, weight_samples, test_canonical=True)

            learned_weights_dict[(feat_size, canonical_action_space_size, complex_action_space_size, task_id, uid)] = (weights, acc)

    weights_labels = list(learned_weights_dict.keys())
    weights_idx = pd.MultiIndex.from_tuples(learned_weights_dict, names=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"])
    weights_set = list(learned_weights_dict.values())
    learned_weights_df = pd.DataFrame(weights_set, index=weights_idx, columns=["learned_weights", "canonical_task_acc"])

    return learned_weights_df

def main(args):

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)