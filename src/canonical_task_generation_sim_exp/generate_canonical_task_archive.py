from dask.distributed import LocalCluster, Client
from typing import Tuple
import pandas as pd

from canonical_task_generation_sim_exp.lib.arguments import parser, args_to_prefix
from canonical_task_generation_sim_exp.canonical_task_search import search

def create_task_archive(dask_client: Client,
                        action_space_range: Tuple = (2, 10),
                        feat_space_range: Tuple = (3, 5),
                        weight_space: str = "normal",
                        metric: str = "dispersion",
                        num_sampled_tasks: int = 10,
                        num_sampled_agents: int = 10,
                        max_experiment_len: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    best_found_tasks = {}
    random_found_tasks = {}
    worst_found_tasks = {}

    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        for a in range(action_space_range[0], action_space_range[1] + 1):
            result = search.find_tasks(dask_client=dask_client,
                                        action_space_size=a,
                                        feat_space_size=f,
                                        weight_space=weight_space,
                                        metric=metric,
                                        num_sampled_tasks=num_sampled_tasks,
                                        num_sampled_agents=num_sampled_agents,
                                        max_experiment_len=max_experiment_len)
            best_found_tasks[(f, a)] = result.best
            random_found_tasks[(f, a)] = result.random
            worst_found_tasks[(f, a)] = result.worst



    best_task_labels = list(best_found_tasks.keys())
    best_task_idx = pd.MultiIndex.from_tuples(best_task_labels, names=["feat_dim", "num_actions"])
    best_tasks = [[t.features, t.preconditions, t.score]for t in best_found_tasks.values()]

    best_df = pd.DataFrame(best_tasks, index=best_task_idx, columns=["features", "preconditions", "score"])

    random_task_labels = list(random_found_tasks.keys())
    random_task_idx = pd.MultiIndex.from_tuples(random_task_labels, names=["feat_dim", "num_actions"])
    random_tasks = [[t.features, t.preconditions, t.score]for t in random_found_tasks.values()]

    random_df = pd.DataFrame(random_tasks, index=random_task_idx, columns=["features", "preconditions", "score"])

    worst_task_labels = list(worst_found_tasks.keys())
    worst_task_idx = pd.MultiIndex.from_tuples(worst_task_labels, names=["feat_dim", "num_actions"])
    worst_tasks = [[t.features, t.preconditions, t.score]for t in worst_found_tasks.values()]

    worst_df = pd.DataFrame(worst_tasks, index=worst_task_idx, columns=["features", "preconditions", "score"])

    return (best_df, random_df, worst_df)

def vis_score_by_action_space_size(task_archive: pd.DataFrame,
                                   feat_space_size: int = 3):

    pass

def main(args):
    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)

    best_task_df, random_task_df, worst_task_df = create_task_archive(dask_client=client,
                                                                        action_space_range=(2, args.max_action_space_size),
                                                                        feat_space_range=(3, args.max_feature_space_size),
                                                                        weight_space=args.weight_space,
                                                                        metric=args.metric,
                                                                        num_sampled_tasks=args.num_experiments,
                                                                        num_sampled_agents=args.weight_samples,
                                                                        max_experiment_len=args.max_experiment_len)

    print(best_task_df, random_task_df, worst_task_df)

    import pathlib

    p = pathlib.Path(f"results/canonical_task_archive/{args_to_prefix(args)}/")
    p.mkdir(parents=True, exist_ok=True)

    with (p / "best_task_archive.csv").open("w") as f:
        best_task_df.to_csv(f)

    with (p / "random_task_archive.csv").open("w") as f:
        random_task_df.to_csv(f)

    with (p / "worst_task_archive.csv").open("w") as f:
        worst_task_df.to_csv(f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)