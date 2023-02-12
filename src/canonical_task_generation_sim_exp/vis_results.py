import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS

def vis_acc(best_task_avg_acc: pd.DataFrame,
            random_task_avg_acc: pd.DataFrame,
            worst_task_avg_acc: pd.DataFrame,
            args) -> None:

    best_task_avg_acc_across_tasks = best_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    b_data = best_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")
    print(b_data)

    random_task_avg_acc_across_tasks = random_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    random_task_avg_acc_across_tasks.index = random_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    r_data = random_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")
    print(r_data)

    worst_task_avg_acc_across_tasks = worst_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")
    print(w_data)

    f, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    ax = axes.flat

    plot = sns.heatmap(
        data=b_data,
        annot=True,
        vmin=.9,
        vmax=1.0,
        ax=ax[0]
    )
    plot.set(title=f"Avg. Aacuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name}) over {args.weight_samples} sampled agents")

    plot = sns.heatmap(
        data=r_data,
        annot=True,
        vmin=.9,
        vmax=1.0,
        ax=ax[1]
    )
    plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks selected at random")

    plot = sns.heatmap(
        data=w_data,
        annot=True,
        vmin=.9,
        vmax=1.0,
        ax=ax[2]
    )
    plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name}) over {args.weight_samples} sampled agents")

    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}.png")

    if not args.headless:
        plt.show()



def main(args):

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)