import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import ComplexTask
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS
from canonical_task_generation_sim_exp.lib.vi import value_iteration
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
        transfer_rewards_abstract = complex_features.dot(transferred_weight)

        # compute policy for transferred rewards
        qf_transfer, _, _ = value_iteration(complex_task.states, complex_task.actions, complex_task.transition, transfer_rewards_abstract,
                                            complex_task.terminal_idx)

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
    print(f" Avg: {predict_score}")

    print("\n")
    print("Complex task:")
    print("   demonstration -", complex_user_demo)
    print("     predictions -", predict_sequence)


    return predict_sequence, predict_score

def vis_acc(best_task_acc: pd.DataFrame,
            random_task_acc: pd.DataFrame,
            worst_task_acc: pd.DataFrame,
            args) -> None:

    f, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    ax = axes.flat

    best_task_acc.index = best_task_acc.index.set_names(["Feature Dimension", "Number of Actions"])
    plot = sns.scatterplot(
        data=best_task_acc,
        x="Number of Actions",
        y="Feature Dimension",
        hue="score",
        size="score",
        hue_norm=(0, 5),
        size_norm=(0, 5),
        ax=ax[0]
    )
    plot.set(title=f"Distingushable reward function metric ({METRICS[args.metric].name}) for best found tasks over {args.weight_samples} sampled agents")


    random_task_archive.index = random_task_archive.index.set_names(["Feature Dimension", "Number of Actions"])
    plot = sns.scatterplot(
        data=random_task_archive,
        x="Number of Actions",
        y="Feature Dimension",
        hue="score",
        size="score",
        hue_norm=(0, 5),
        size_norm=(0, 5),
        ax=ax[1]
    )
    plot.set(title=f"Distingushable reward function metric ({METRICS[args.metric].name}) for randomly selected tasks over {args.weight_samples} sampled agents")

    worst_task_archive.index = worst_task_archive.index.set_names(["Feature Dimension", "Number of Actions"])
    plot = sns.scatterplot(
        data=worst_task_archive,
        x="Number of Actions",
        y="Feature Dimension",
        hue="score",
        size="score",
        hue_norm=(0, 5),
        size_norm=(0, 5),
        ax=ax[2]
    )
    plot.set(title=f"Distingushable reward function metric ({METRICS[args.metric].name}) for worst found tasks over {args.weight_samples} sampled agents")

    p = out_path(args, kind="figures", owner="canonical_task_archive")
    plt.savefig(p / f"reward_function_metric_sampled_agents{args.weight_samples}.png")

    if not args.headless:
        plt.show()

def save_eval_results(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="results", owner="learned_rf_acc")

    with (p / f"{kind}_learned_rf_acc.csv").open("w") as f:
        task_df.to_csv(f)

def eval(complex_task_archive: pd.DataFrame,
         learned_rf_weights: pd.DataFrame,
         user_complex_demos: pd.DataFrame,
         feat_size: int,
         canonical_action_space_size: int,
         complex_action_space_size: int) -> pd.DataFrame:

    complex_task_set = complex_task_archive.xs((feat_size, complex_action_space_size), level=["feat_dim", "num_actions"])
    user_demo_set = user_complex_demos.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])
    learned_weights_set = learned_rf_weights.xs((feat_size, canonical_action_space_size, complex_action_space_size), level=["feat_dim", "num_canonical_actions", "num_complex_actions"])

    rf_acc = {}
    for ((task_id, task_demo_df),(_, task_learned_weights)) in zip(user_demo_set.groupby(level=["complex_task_id"]), learned_weights_set.groupby(level=["complex_task_id"])):
        print("+++++++++++++++++++++++++++++")
        print("Task:", task_id)
        complex_task_info = complex_task_set.iloc[task_id]
        complex_task = ComplexTask(complex_task_info["features"], complex_task_info["preconditions"])
        complex_task.set_end_state(list(range(len(complex_task_info["features"]))))
        complex_task.enumerate_states()
        complex_task.set_terminal_idx()

        for ((uid, user_task_demos),(_, user_task_learned_weights)) in zip(task_demo_df.groupby(level=["uid"]), task_learned_weights.groupby(level=["uid"])):
            print("=======================")
            print("User:", uid)
            complex_demo = user_task_demos.loc[(task_id, uid)]["complex_demo"]
            weights = user_task_learned_weights.loc[(task_id, uid)]["learned_weights"]

            pred_demo, acc = evaluate_rf_acc(complex_task, weights, complex_demo)

            rf_acc[(feat_size, canonical_action_space_size, complex_action_space_size, task_id, uid)] = (pred_demo, acc)

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