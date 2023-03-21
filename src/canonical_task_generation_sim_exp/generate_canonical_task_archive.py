from dask.distributed import LocalCluster, Client
from typing import Tuple, Any
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from canonical_task_generation_sim_exp.lib.arguments import parser, args_to_prefix, out_path
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.canonical_task_search import search
from canonical_task_generation_sim_exp.canonical_task_search.metrics import METRICS

def create_canonical_task_archive(dask_client: Client,
                                user_archive: pd.DataFrame,
                                action_space_range: Tuple = (2, 10),
                                feat_space_range: Tuple = (3, 5),
                                weight_space: str = "normal",
                                metric: str = "dispersion",
                                num_sampled_tasks: int = 10,
                                num_sampled_agents: int = 10,
                                max_experiment_len: int = 100,
                                num_random_tasks: int = 10,
                                args: Any = None) -> pd.DataFrame:

    found_tasks = {}

    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        feat_user_df = user_archive.loc[[f]]
        feat_users = feat_user_df["users"]
        for a in range(action_space_range[0], action_space_range[1] + 1):
            result = search.find_tasks(dask_client=dask_client,
                                        agent_archive=feat_users,
                                        action_space_size=a,
                                        feat_space_size=f,
                                        weight_space=weight_space,
                                        metric=metric,
                                        num_sampled_tasks=num_sampled_tasks,
                                        num_sampled_agents=num_sampled_agents,
                                        max_experiment_len=max_experiment_len,
                                        num_random_tasks=num_random_tasks,
                                        args=args)

            for i, task in enumerate(result.tasks):
                found_tasks[(f, a, i)] = task


    task_labels = list(found_tasks.keys())
    task_idx = pd.MultiIndex.from_tuples(task_labels, names=["feat_dim", "num_actions", "id"])
    tasks = [[t.features, t.preconditions, t.score, t.kind] for t in found_tasks.values()]

    task_df = pd.DataFrame(tasks, index=task_idx, columns=["features", "preconditions", "score", "kind"])

    return task_df

def create_score_spanning_canonical_task_archive(dask_client: Client,
                                                user_archive: pd.DataFrame,
                                                action_space_range: Tuple = (2, 10),
                                                feat_space_range: Tuple = (3, 5),
                                                weight_space: str = "normal",
                                                metric: str = "dispersion",
                                                num_spanning_tasks: int = 10,
                                                num_sampled_tasks: int = 10,
                                                num_sampled_agents: int = 10,
                                                max_experiment_len: int = 100,
                                                args: Any = None) -> pd.DataFrame:

    found_tasks = {}


    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        feat_user_df = user_archive.loc[[f]]
        feat_users = feat_user_df["users"]
        for a in range(action_space_range[0], action_space_range[1] + 1):
            result = search.find_tasks_spanning_metric(dask_client=dask_client,
                                                        agent_archive=feat_users,
                                                        action_space_size=a,
                                                        feat_space_size=f,
                                                        weight_space=weight_space,
                                                        metric=metric,
                                                        num_sampled_tasks=num_sampled_tasks,
                                                        num_sampled_agents=num_sampled_agents,
                                                        max_experiment_len=max_experiment_len,
                                                        num_results=num_spanning_tasks,
                                                        args=args)
            for i, task in enumerate(result.tasks):
                found_tasks[(f, a, i)] = task


    task_labels = list(found_tasks.keys())
    task_idx = pd.MultiIndex.from_tuples(task_labels, names=["feat_dim", "num_actions", "id"])
    tasks = [[t.features, t.preconditions, t.score] for t in found_tasks.values()]

    found_task_df = pd.DataFrame(tasks, index=task_idx, columns=["features", "preconditions", "score"])

    return found_task_df

def vis_score_by_action_space_size(best_task_archive: pd.DataFrame,
                                   random_task_archive: pd.DataFrame,
                                   worst_task_archive: pd.DataFrame,
                                   args,
                                   feat_space_size: int = 3) -> None:

    sns.set(rc={"figure.figsize": (20, 10)})

    best_task_by_action = best_task_archive.xs(feat_space_size, level="feat_dim")
    random_task_by_action = random_task_archive.xs(feat_space_size, level="feat_dim")
    worst_task_by_action = worst_task_archive.xs(feat_space_size, level="feat_dim")

    scores = pd.concat([best_task_by_action.score, random_task_by_action.score, worst_task_by_action.score], axis=1)
    scores.columns = [
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Best Task for Action Space Size N",
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Random Task for Action Space Size N",
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Worst Task for Action Space Size N"
    ]

    scores.index.name = "Action Space Size"

    plot = sns.lineplot(data=scores)
    plot.set(title=f"Action space vs. distingushable reward function metric ({METRICS[args.metric].name}) for {args.weight_samples} sampled agents (feature space size={feat_space_size}, sampled tasks={args.num_experiments})")
    p = out_path(args, kind="figures", owner="canonical_task_archive")
    plt.savefig(p / f"action_vs_metric_score_feat_dim{feat_space_size}.png")

    if not args.headless:
        plt.show()

    plt.close()

def vis_score_by_feat_space_size(best_task_archive: pd.DataFrame,
                                   random_task_archive: pd.DataFrame,
                                   worst_task_archive: pd.DataFrame,
                                   args,
                                   action_space_size: int = 3) -> None:

    sns.set(rc={"figure.figsize": (20, 10)})

    best_task_by_action = best_task_archive.xs(action_space_size, level="num_actions")
    random_task_by_action = random_task_archive.xs(action_space_size, level="num_actions")
    worst_task_by_action = worst_task_archive.xs(action_space_size, level="num_actions")

    scores = pd.concat([best_task_by_action.score, random_task_by_action.score, worst_task_by_action.score], axis=1)
    scores.columns = [
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Best Task for Feature Space Size N",
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Random Task for Feature Space Size N",
        f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Worst Task for Feature Space Size N"
    ]

    scores.index.name = "Feature Space Size"

    plot = sns.lineplot(data=scores)
    plot.set(title=f"Feature space size vs. distingushable reward function metric ({METRICS[args.metric].name}) for {args.weight_samples} sampled agents (action space size={action_space_size}, sampled tasks={args.num_experiments})")
    p = out_path(args, kind="figures", owner="canonical_task_archive")
    plt.savefig(p / f"action_vs_metric_score_action_space_size{action_space_size}.png")

    if not args.headless:
        plt.show()

    plt.close()

def vis_score(best_task_archive: pd.DataFrame,
              random_task_archive: pd.DataFrame,
              worst_task_archive: pd.DataFrame,
              args) -> None:

    f, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    ax = axes.flat

    best_task_archive.index = best_task_archive.index.set_names(["Feature Dimension", "Number of Actions"])
    plot = sns.scatterplot(
        data=best_task_archive,
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


def load_tasks(kind: str, args) -> pd.DataFrame:
    import numpy as np

    p = out_path(args, kind="data", owner="canonical_task_archive", load=True)
    task_df = pd.read_csv(p / f"{kind}_task_archive.csv", index_col=[0,1], converters={'features': serialization.from_list, 'preconditions': serialization.from_np_array})
    task_df["features"] = task_df["features"].apply(np.array)
    task_df["preconditions"] = task_df["preconditions"].apply(np.array)
    return task_df

def load_score_span_tasks(kind: str, args) -> pd.DataFrame:
    import numpy as np

    p = out_path(args, kind="data", owner="canonical_task_archive", load=True)
    task_df = pd.read_csv(p / f"{kind}_task_archive.csv", index_col=[0,1,2], converters={'features': serialization.from_list, 'preconditions': serialization.from_np_array})
    task_df["features"] = task_df["features"].apply(np.array)
    task_df["preconditions"] = task_df["preconditions"].apply(np.array)
    return task_df

def save_tasks(kind: str, task_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="canonical_task_archive")

    with (p / f"{kind}_task_archive.csv").open("w") as f:
        task_df.to_csv(f)


def main(args):
    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)

    best_task_df, random_task_df, worst_task_df = create_canonical_task_archive(dask_client=client,
                                                                                action_space_range=(2, args.max_canonical_action_space_size),
                                                                                feat_space_range=(3, args.max_feature_space_size),
                                                                                weight_space=args.weight_space,
                                                                                metric=args.metric,
                                                                                num_sampled_tasks=args.num_experiments,
                                                                                num_sampled_agents=args.weight_samples,
                                                                                max_experiment_len=args.max_experiment_len)

    save_tasks("best", best_task_df, args)
    save_tasks("random", random_task_df, args)
    save_tasks("worst", worst_task_df, args)

    vis_score_by_action_space_size(best_task_df, random_task_df, worst_task_df, args=args)
    vis_score_by_feat_space_size(best_task_df, random_task_df, worst_task_df, args=args)
    vis_score(best_task_df, random_task_df, worst_task_df, args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)