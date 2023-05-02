import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS

sns.set(font_scale=2, rc={'figure.figsize':(15,9)})

def vis_complex_acc_for_feat(best_task_avg_acc: pd.DataFrame,
                            random_task_avg_acc: pd.DataFrame,
                            worst_task_avg_acc: pd.DataFrame,
                            feat_dim: int,
                            args) -> None:

    best_df = best_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    best_df["num_canonical_actions"] = best_df.num_canonical_actions.astype(str)
    best_df["Number of Canonical Actions"] = "best " + best_df.num_canonical_actions
    best_df = best_df.drop(columns=["num_canonical_actions"])

    plot = sns.lineplot(
        data=best_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=1, rot=-.75, hue=1)
    )

    random_df = random_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    random_df["num_canonical_actions"] = random_df.num_canonical_actions.astype(str)
    random_df["Number of Canonical Actions"] = "random " + random_df.num_canonical_actions
    random_df = random_df.drop(columns=["num_canonical_actions"])

    plot = sns.lineplot(
        data=random_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=2, rot=-.75, hue=1)
    )

    worst_df = worst_task_avg_acc.xs([feat_dim], level=["feat_dim"]).reset_index()
    worst_df["num_canonical_actions"] = worst_df.num_canonical_actions.astype(str)
    worst_df["Number of Canonical Actions"] = "worst " + worst_df.num_canonical_actions
    worst_df = worst_df.drop(columns=["num_canonical_actions"])

    plot = sns.lineplot(
        data=worst_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=3, rot=-.75, hue=1)
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

    plot = sns.lineplot(
        data=best_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Feature Size / Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=1, rot=-.75, hue=1)
    )

    random_df = random_task_avg_acc.reset_index()
    random_df["feat_dim"] = random_df.feat_dim.astype(str)
    random_df["num_canonical_actions"] = random_df.num_canonical_actions.astype(str)
    random_df["Feature Size / Number of Canonical Actions"] = "random " + random_df.feat_dim.str.cat(random_df.num_canonical_actions, sep="/")
    random_df = random_df.drop(columns=["feat_dim", "num_canonical_actions"])

    plot = sns.lineplot(
        data=random_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Feature Size / Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=2, rot=-.75, hue=1)
    )

    worst_df = worst_task_avg_acc.reset_index()
    worst_df["feat_dim"] = worst_df.feat_dim.astype(str)
    worst_df["num_canonical_actions"] = worst_df.num_canonical_actions.astype(str)
    worst_df["Feature Size / Number of Canonical Actions"] = "worst " + worst_df.feat_dim.str.cat(worst_df.num_canonical_actions, sep="/")
    worst_df = worst_df.drop(columns=["feat_dim", "num_canonical_actions"])

    plot = sns.lineplot(
        data=worst_df,
        x="num_complex_actions",
        y="complex_task_acc",
        hue="Feature Size / Number of Canonical Actions",
        palette=sns.cubehelix_palette(start=3, rot=-.75, hue=1)
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
    print(b_data)

    random_task_avg_acc_across_tasks = random_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    random_task_avg_acc_across_tasks.index = random_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    r_data = random_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    #f, axes = plt.subplots(3, 1, figsize=(10, 24), sharex=True, sharey=True)
    #ax = axes.flat

    plot = sns.heatmap(
        data=b_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[0]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name})")
    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_best.png")

    if not args.headless:
        plt.show()

    plt.clf()

    plot = sns.heatmap(
        data=r_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[1]
    )
    #plot.set(title="Avg. accuracy on complex tasks using canonical tasks selected at random")
    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_random.png")

    if not args.headless:
        plt.show()

    plt.clf()

    plot = sns.heatmap(
        data=w_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[2]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name})")

    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_worst.png")

    if not args.headless:
        plt.show()


def vis_feat_acc(best_task_avg_acc: pd.DataFrame,
            random_task_avg_acc: pd.DataFrame,
            worst_task_avg_acc: pd.DataFrame,
            args) -> None:

    best_task_avg_acc_across_tasks = best_task_avg_acc.groupby(level=["num_complex_actions", "num_canonical_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Number of Actions in Complex Task", "Number of Actions in Canonical Task",])
    b_data = best_task_avg_acc_across_tasks.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")
    print(b_data)

    random_task_avg_acc_across_tasks = random_task_avg_acc.groupby(level=["num_complex_actions", "num_canonical_actions"]).mean()
    random_task_avg_acc_across_tasks.index = random_task_avg_acc_across_tasks.index.set_names(["Number of Actions in Complex Task", "Number of Actions in Canonical Task"])
    r_data = random_task_avg_acc_across_tasks.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_task_avg_acc.groupby(level=["num_complex_actions", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Number of Actions in Complex Task", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

    #f, axes = plt.subplots(3, 1, figsize=(10, 24), sharex=True, sharey=True)
    #ax = axes.flat

    plot = sns.heatmap(
        data=b_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[0]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name})")
    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_feat_best.png")

    if not args.headless:
        plt.show()

    plt.clf()

    plot = sns.heatmap(
        data=r_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[1]
    )
    #plot.set(title="Avg. accuracy on complex tasks using canonical tasks selected at random")
    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_feat_random.png")

    if not args.headless:
        plt.show()

    plt.clf()

    plot = sns.heatmap(
        data=w_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[2]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name})")

    p = out_path(args, kind="figures", owner="accuracy")
    plt.savefig(p / f"reward_acc{args.weight_samples}_feat_worst.png")

    if not args.headless:
        plt.show()

def vis_con_comp_acc(best_task_avg_acc: pd.DataFrame,
            random_task_avg_acc: pd.DataFrame,
            worst_task_avg_acc: pd.DataFrame,
            args) -> None:

    best_task_avg_acc_across_tasks = best_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task"])
    best_data = best_task_avg_acc_across_tasks.reset_index()#.pivot("Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task", "complex_task_acc")

    random_task_avg_acc_across_tasks = random_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions"]).mean()
    random_task_avg_acc_across_tasks.index = random_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task"])
    random_data = random_task_avg_acc_across_tasks.reset_index()#.pivot("Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_task_avg_acc.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task"])
    worst_data = worst_task_avg_acc_across_tasks.reset_index()#.pivot(index=["Feature Dimension", "Number of Actions in Canonical Task", "Number of Actions in Complex Task", "complex_task_acc"])

    #f, axes = plt.subplots(3, 1, figsize=(10, 24), sharex=True, sharey=True)
    #ax = axes.flat

    for f in range(2, args.max_feature_space_size+1):
        b_data = best_data[best_data["Feature Dimension"] == f]
        print(b_data)
        b_data = b_data.drop(columns=["Feature Dimension"])
        b_data = b_data.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

        r_data = random_data[random_data["Feature Dimension"] == f]
        r_data = r_data.drop(columns=["Feature Dimension"])
        r_data = r_data.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

        w_data = worst_data[worst_data["Feature Dimension"] == f]
        w_data = w_data.drop(columns=["Feature Dimension"])
        w_data = w_data.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

        plot = sns.heatmap(
            data=b_data,
            annot=True,
            vmin=.8,
            vmax=1.,
            cmap=sns.color_palette("viridis", as_cmap=True),
            annot_kws={"size": 30}
            #ax=ax[0]
        )
        #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name})")
        p = out_path(args, kind="figures", owner="accuracy")
        plt.savefig(p / f"reward_acc{args.weight_samples}_feat_{f}_best.png")

        if not args.headless:
            plt.show()

        plt.clf()

        plot = sns.heatmap(
            data=r_data,
            annot=True,
            vmin=.8,
            vmax=1.,
            cmap=sns.color_palette("viridis", as_cmap=True),
            annot_kws={"size": 30}
            #ax=ax[1]
        )
        #plot.set(title="Avg. accuracy on complex tasks using canonical tasks selected at random")
        p = out_path(args, kind="figures", owner="accuracy")
        plt.savefig(p / f"reward_acc{args.weight_samples}_feat_{f}_random.png")

        if not args.headless:
            plt.show()

        plt.clf()

        plot = sns.heatmap(
            data=w_data,
            annot=True,
            vmin=.8,
            vmax=1.,
            cmap=sns.color_palette("viridis", as_cmap=True),
            annot_kws={"size": 30}
            #ax=ax[2]
        )
        #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name})")
        p = out_path(args, kind="figures", owner="accuracy")
        plt.savefig(p / f"reward_acc{args.weight_samples}_feat_{f}_worst.png")

        if not args.headless:
            plt.show()


def vis_score_v_acc(score_spanning_task_acc_df: pd.DataFrame, canonical_task_archive: pd.DataFrame, args) -> None:

    score_df = canonical_task_archive.reset_index()
    score_df["feat_dim"] = score_df.feat_dim.astype(str)
    score_df["num_actions"] = score_df.num_actions.astype(str)
    score_df["Feature Size / Number of Canonical Actions"] = score_df.feat_dim.str.cat(score_df.num_actions, sep="/")
    score_df["num_canonical_actions"] = score_df["num_actions"]
    score_df = score_df.drop(columns=["num_actions", "features", "preconditions", "feat_dim"])

    acc_df = score_spanning_task_acc_df.reset_index()
    acc_df["feat_dim"] = acc_df.feat_dim.astype(str)
    acc_df["num_canonical_actions"] = acc_df.num_canonical_actions.astype(str)
    acc_df["Feature Size / Number of Canonical Actions"] = acc_df.feat_dim.str.cat(acc_df.num_canonical_actions, sep="/")
    acc_df.set_index(["feat_dim", "num_canonical_actions",  "canonical_task_id", "num_complex_actions", "complex_task_id", "uid", "Feature Size / Number of Canonical Actions"], inplace=True)
    acc_df = acc_df.groupby(level=["feat_dim",  "num_canonical_actions",  "canonical_task_id", "num_complex_actions", "Feature Size / Number of Canonical Actions"]).mean()
    acc_df = acc_df.reset_index()
    acc_df = acc_df.drop(columns=["feat_dim",  "num_canonical_actions"])

    feat_as_vals = score_df["Feature Size / Number of Canonical Actions"].unique()

    for val in feat_as_vals:
        fas_score_df = score_df.loc[score_df["Feature Size / Number of Canonical Actions"] == val]
        fas_acc_df = acc_df.loc[acc_df["Feature Size / Number of Canonical Actions"] == val]
        complex_task_sizes = fas_acc_df["num_complex_actions"].unique()

        score_v_acc = {f"Score on metric ({args.metric})":[], "Accuracy":[], "Complex Action Size":[]}
        for com_as in complex_task_sizes:
            com_as_fas_acc_df = fas_acc_df.loc[fas_acc_df["num_complex_actions"] == com_as]

            for id in fas_score_df["id"]:
                score = fas_score_df[fas_score_df["id"] == id]["score"].item()
                acc = com_as_fas_acc_df[com_as_fas_acc_df["canonical_task_id"] == id]["complex_task_acc"].item()
                score_v_acc[f"Score on metric ({args.metric})"].append(score)
                score_v_acc["Accuracy"].append(acc)
                score_v_acc["Complex Action Size"].append(com_as)

        score_v_acc = pd.DataFrame.from_dict(score_v_acc)

        plot = sns.lineplot(
            data=score_v_acc,
            x=f"Score on metric ({args.metric})",
            y="Accuracy",
            hue="Complex Action Size",
            palette=sns.color_palette("Paired")
        )

        plot.set(title=f"Accuracy on complex tasks using canonical tasks (Num. Features / Num. Actions {val})\n of different metric scores ({METRICS[args.metric].name})")


        p = out_path(args, kind="figures", owner="accuracy")
        val_out = val.replace("/","_")
        plt.savefig(p / f"score_v_acc_metric-{args.metric}-feat_can_as_{val_out}.png")

        if not args.headless:
            pass
            #plt.show()

        plt.close()


def main(args):

    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)