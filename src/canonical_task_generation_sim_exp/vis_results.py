import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS

def vis_complex_acc_for_feat(best_task_avg_acc: pd.DataFrame,
                            random_task_avg_acc: pd.DataFrame,
                            worst_task_avg_acc: pd.DataFrame,
                            feat_dim: int,
                            args) -> None:

    best_df = best_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    best_df["num_canonical_actions"] = best_df.num_canonical_actions.astype(str)
    best_df["Number of Canonical Actions"] = "best " + best_df.num_canonical_actions
    best_df = best_df.drop(columns=["num_canonical_actions"])

    random_df = random_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    random_df["num_canonical_actions"] = random_df.num_canonical_actions.astype(str)
    random_df["Number of Canonical Actions"] = "random " + random_df.num_canonical_actions
    random_df = random_df.drop(columns=["num_canonical_actions"])

    worst_df = worst_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    worst_df["num_canonical_actions"] = worst_df.num_canonical_actions.astype(str)
    worst_df["Number of Canonical Actions"] = "worst " + worst_df.num_canonical_actions
    worst_df = worst_df.drop(columns=["num_canonical_actions"])

    result_df = pd.concat([best_df, random_df, worst_df], ignore_index=True)

    plot = sns.lineplot(
        data=result_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Number of Canonical Actions"
    )



    plot.set(title=f"Accuracy on complex tasks (feature dim: {feat_dim}) using canonical tasks\n selected by metric score ({METRICS[args.metric].name}) over {args.weight_samples} sampled agents")

    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"acc_by_abstract_task_{args.weight_samples}_feat{feat_dim}.png")

    if not args.headless:
        plt.show()


def vis_complex_acc(best_task_avg_acc: pd.DataFrame,
                    random_task_avg_acc: pd.DataFrame,
                    worst_task_avg_acc: pd.DataFrame,
                    args) -> None:

    best_df = best_task_avg_acc.reset_index()
    best_df["feat_dim"] = best_df.feat_dim.astype(str)
    best_df["num_canonical_actions"] = best_df.num_canonical_actions.astype(str)
    best_df["Feature Size / Number of Canonical Actions"] = "best " + best_df.feat_dim.str.cat(best_df.num_canonical_actions, sep="/")
    best_df = best_df.drop(columns=["feat_dim", "num_canonical_actions"])

    random_df = random_task_avg_acc.reset_index()
    random_df["feat_dim"] = random_df.feat_dim.astype(str)
    random_df["num_canonical_actions"] = random_df.num_canonical_actions.astype(str)
    random_df["Feature Size / Number of Canonical Actions"] = "random " + random_df.feat_dim.str.cat(random_df.num_canonical_actions, sep="/")
    random_df = random_df.drop(columns=["feat_dim", "num_canonical_actions"])

    worst_df = worst_task_avg_acc.reset_index()
    worst_df["feat_dim"] = worst_df.feat_dim.astype(str)
    worst_df["num_canonical_actions"] = worst_df.num_canonical_actions.astype(str)
    worst_df["Feature Size / Number of Canonical Actions"] = "worst " + worst_df.feat_dim.str.cat(worst_df.num_canonical_actions, sep="/")
    worst_df = worst_df.drop(columns=["feat_dim", "num_canonical_actions"])

    result_df = pd.concat([best_df, random_df, worst_df], ignore_index=True)

    plot = sns.lineplot(
        data=result_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Feature Size / Number of Canonical Actions"
    )



    plot.set(title=f"Accuracy on complex tasks using canonical tasks\n selected by metric score ({METRICS[args.metric].name}) over {args.weight_samples} sampled agents")

    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"acc_by_abstract_task_{args.weight_samples}.png")

    if not args.headless:
        plt.show()


def vis_avg_acc(best_task_avg_acc: pd.DataFrame,
            random_task_avg_acc: pd.DataFrame,
            worst_task_avg_acc: pd.DataFrame,
            args) -> None:

    best_task_avg_acc_across_tasks = best_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    b_data = best_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    random_task_avg_acc_across_tasks = random_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    random_task_avg_acc_across_tasks.index = random_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    r_data = random_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

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