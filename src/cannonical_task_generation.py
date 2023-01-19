import subprocess
import argparse

parser = argparse.ArgumentParser(description='Repeated IRL experiements')
parser.add_argument('--num-experiments', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--weight-samples', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--max-action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions')
parser.add_argument('--max-feature-space-size', type=int,
                    default=4,
                    help='Dimensionality of feature space describing actions')
parser.add_argument('--max-experiment-len', type=int,
                    default=100,
                    help='Maximum number of steps taken in each experiment')
parser.add_argument("--verbose", action='store_true', help='Print selected tasks')
parser.add_argument("--load-from-file", type=str, help="Load a task from a saved file")
parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes to run experiements")
parser.add_argument("--metric", type=str, default="unique-trajectories",
                    help="What metric to use to determine if a task is good a distingushing reward functions")
parser.add_argument("--weight-space", type=str, default="normal", help="What space to sample weights from")
parser.add_argument("--iterations", type=int, default=1, help="What iteration of the experiment")
parser.add_argument("--headless", action='store_true', help='Dont show figures')

args = parser.parse_args()

MAX_FEAT_RANGE=5

for f in range(3, args.max_feature_space_size+1):
    for i in range(args.iterations):
        subp_args = [
            "--num-experiments", args.num_experiments,
            "--max-action-space-size", args.max_action_space_size,
            "--feature-space-size", f,
            "--weight-samples", args.weight_samples,
            "--num-workers", args.num_workers,
            "--metric", args.metric,
            "--weight-space", args.weight_space,
            "--iteration", i + 1,
            "--headless"
        ]
        subp_args = [str(a) for a in subp_args]
        rirl_args = ["python3", "rirl"] + subp_args
        print(rirl_args)
        subprocess.run(rirl_args)

        simulate_users_args = ["python3", "simulate_users.py"] + subp_args
        print(simulate_users_args)
        subprocess.run(simulate_users_args)

        experiment_sim_args = ["python3", "experiments_sim.py"] + subp_args
        print(experiment_sim_args)
        subprocess.run(experiment_sim_args)

    FILE_SUFFIX = f"exp{args.num_experiments}_feat{f}_metric_{args.metric}_space_{args.weight_space}"

    import pickle

    mean_accuracies = {}
    for i in range(args.iterations):
        with open(f"results/complex_prediction_accuracy_{FILE_SUFFIX}_{i}.pkl", "rb") as f:
            run_accuracies = pickle.load(f)
        for t in ["best", "random", "worst"]:
            if t not in list(mean_accuracies.keys()):
                mean_accuracies[t] = run_accuracies[t]
            else:
                acc = {}
                for (m, r) in zip(mean_accuracies[t].items(), run_accuracies[t].items()):
                    acc[m[0]] = m[1] + r[1]
                print(acc)
                mean_accuracies[t] = acc
                print(mean_accuracies)

    for t in ["best", "random", "worst"]:
        for j, m in mean_accuracies[t].items():
            mean_accuracies[t][j] = m / args.iterations

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(mean_accuracies)

    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns

    best_mean_accuracies = mean_accuracies["best"]
    random_mean_accuracies = mean_accuracies["random"]
    worst_mean_accuracies = mean_accuracies["worst"]

    metric_df = pd.DataFrame({
        "Action Space Size": list(best_mean_accuracies.keys()),
        f"Prediction accuracy on Complex Task \nfor using Best Cannonical Task for Action Space Size N ({args.iterations} iterations)": list(best_mean_accuracies.values()),
        f"Prediction accuracy on Complex Task \nfor using Random Cannonical Task for Action Space Size N ({args.iterations} iterations)": list(random_mean_accuracies.values()),
        f"Prediction accuracy on Complex \nfor using Worst Cannonical Task for Action Space Size N ({args.iterations} iterations)": list(worst_mean_accuracies.values())
    })
    metric_df.name = f"Action Space Size vs. MaxEnt Prediction Accuracy"

    plot = sns.lineplot(x="Action Space Size",
                    y=f"Prediction accuracy on complex task",
                    hue="Task Class",
                    data=pd.melt(metric_df,
                                ["Action Space Size"],
                                var_name="Task Class",
                                value_name=f"Prediction accuracy on complex task"))
    plot.set(title=f"Action space vs. Prediction accuracy on complex task")
    plot.set_ylim(0.45, 1.0)
    plt.savefig(f"figures/action_space_vs_complex_prediction_acc_feat_space_size_{FILE_SUFFIX}_iterations_{args.iterations}.png")
#plt.show()

