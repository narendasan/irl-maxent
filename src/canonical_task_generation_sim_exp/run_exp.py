import subprocess
import argparse
import pandas as pd
from rich.progress import track
from dask.distributed import LocalCluster, Client

from canonical_task_generation_sim_exp.lib.arguments import parser
from canonical_task_generation_sim_exp.lib.action_space_range import complex_action_space_range
from canonical_task_generation_sim_exp import simulate_user_demos
from canonical_task_generation_sim_exp.generate_canonical_task_archive import create_canonical_task_archive
from canonical_task_generation_sim_exp.generate_canonical_task_archive import save_tasks as save_canonical_tasks
from canonical_task_generation_sim_exp.generate_canonical_task_archive import load_score_span_tasks as load_canonical_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import create_complex_task_archive
from canonical_task_generation_sim_exp.generate_complex_task_archive import save_tasks as save_complex_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import load_tasks as load_complex_tasks
from canonical_task_generation_sim_exp.sample_users import save_users, load_users, create_user_archive
from canonical_task_generation_sim_exp.learn_reward_function import train, save_learned_weights, load_learned_weights
from canonical_task_generation_sim_exp.evaluate_results import eval, save_eval_results, load_eval_results, avg_complex_task_acc, save_processed_results, load_processed_results
from canonical_task_generation_sim_exp.vis_results import vis_score_v_acc
from canonical_task_generation_sim_exp.vis_results import vis_avg_acc, vis_complex_acc, vis_complex_acc_for_feat


def main(args):

    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)


    if args.load_canonical_tasks:
        canonical_tasks_archive = load_canonical_tasks("search_results", args)
    else:
        canonical_tasks_archive = create_canonical_task_archive(dask_client=client,
                                                                action_space_range=(2, args.max_canonical_action_space_size),
                                                                feat_space_range=(2, args.max_feature_space_size),
                                                                weight_space=args.weight_space,
                                                                metric=args.metric,
                                                                num_sampled_tasks=args.num_experiments,
                                                                num_sampled_agents=args.weight_samples,
                                                                max_experiment_len=args.max_experiment_len,
                                                                args=args)

        save_canonical_tasks("search_results", canonical_tasks_archive, args)

    if not (args.load_results or args.load_predictions):
        if args.load_complex_tasks:
            complex_tasks_archive = load_complex_tasks(args)
        else:
            # Actual task sizes now start from max canonical size and go to max complex size
            complex_tasks_archive = create_complex_task_archive(action_space_range=(args.max_canonical_action_space_size, args.max_complex_action_space_size),
                                                                feat_space_range=(2, args.max_feature_space_size),
                                                                num_tasks_per_quadrant=args.num_test_tasks)

            save_complex_tasks(complex_tasks_archive, args)

        if args.load_test_users:
            users = load_users(args)
        else:
            users = create_user_archive(feat_space_range=(2, args.max_feature_space_size), num_users=args.num_test_users, weight_space=args.weight_space)

            save_users(users, args)

        if args.load_user_demos:
            demos_df = simulate_user_demos.load_demos("search_results", args)
        else:
            demos_df = None

            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        can_task_score = can_task["score"].item()
                        for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Simulate user demos (can_id: {can_task_id} score: {can_task_score}) - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")
                            demo_df = simulate_user_demos.sim_demos(client, can_task, complex_tasks_archive, feat_users, f, canonical_as, complex_as)
                            demo_df['canonical_task_id'] = can_task_id
                            demo_df.set_index('canonical_task_id', append=True, inplace=True)
                            demo_df.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if demos_df is None:
                                demos_df = demo_df
                            else:
                                demos_df = pd.concat([demos_df, demo_df])

            simulate_user_demos.save_demos("search_results", demos_df, args)

        if args.load_learned_user_rfs and args.load_user_demos:
            learned_weights_df = load_learned_weights("search_results", args)
        elif args.load_learned_user_rfs:
            raise RuntimeError("To load already learned rfs, user demos must also be loaded")
        else:
            learned_weights_df = None
            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    demo_as_df = demos_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        demo = demo_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        can_task_score = can_task["score"].item()
                        for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Learn reward function (can_id: {can_task_id} score: {can_task_score}) - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                            learned_weights = train(client, can_task, complex_tasks_archive, demo, f, canonical_as, complex_as)
                            learned_weights['canonical_task_id'] = can_task_id
                            learned_weights.set_index('canonical_task_id', append=True, inplace=True)
                            learned_weights.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if learned_weights_df is None:
                                learned_weights_df = learned_weights
                            else:
                                learned_weights_df = pd.concat([learned_weights_df, learned_weights])

            save_learned_weights("search_results", learned_weights_df, args)

    if not args.load_results:
        if args.load_predictions:
            task_acc_df = load_eval_results("search_results", args)
        else:
            task_acc_df = None
            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    demo_as_df = demos_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    weights_as_df = learned_weights_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    can_task_score = can_task["score"].item()
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        demo = demo_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        weights = weights_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Learn reward function (can_id: {can_task_id} score: {can_task_score}) - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                            task_acc = eval(client, complex_tasks_archive, weights, demo, f, canonical_as, complex_as)
                            task_acc['canonical_task_id'] = can_task_id
                            task_acc.set_index('canonical_task_id', append=True, inplace=True)
                            task_acc.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if task_acc_df is None:
                                task_acc_df = task_acc
                            else:
                                task_acc_df = pd.concat([task_acc_df, task_acc])

        save_eval_results("search_results", task_acc_df, args)

        cleaned_task_acc_df = task_acc_df.drop(columns=["predicted_complex_demo"])

        task_archive = canonical_tasks_archive.reset_index()
        best_task_acc_df = cleaned_task_acc_df[cleaned_task_acc_df.index.isin(task_archive[task_archive["kind"] == "best"]["id"], level="canonical_task_id")]
        random_task_acc_df =  cleaned_task_acc_df[cleaned_task_acc_df.index.isin(task_archive[task_archive["kind"] == "random"]["id"], level="canonical_task_id")]
        worst_task_acc_df =  cleaned_task_acc_df[cleaned_task_acc_df.index.isin(task_archive[task_archive["kind"] == "worst"]["id"], level="canonical_task_id")]

        best_complex_task_acc_df = best_task_acc_df.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"]).mean()
        random_complex_task_acc_df = random_task_acc_df.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"]).mean()
        worst_complex_task_acc_df = worst_task_acc_df.groupby(level=["feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_id", "uid"]).mean()

        save_eval_results("best", best_complex_task_acc_df, args)
        save_eval_results("random", random_complex_task_acc_df, args)
        save_eval_results("worst", worst_complex_task_acc_df, args)

    if args.load_results:
        best_avg_acc_df = load_processed_results("best", args)
        random_avg_acc_df = load_processed_results("random", args)
        worst_avg_acc_df = load_processed_results("worst", args)
    else:
        best_avg_acc_df = avg_complex_task_acc(best_complex_task_acc_df)
        random_avg_acc_df = avg_complex_task_acc(random_complex_task_acc_df)
        worst_avg_acc_df = avg_complex_task_acc(worst_complex_task_acc_df)

        save_processed_results("best", best_avg_acc_df, args)
        save_processed_results("random", random_avg_acc_df, args)
        save_processed_results("worst", worst_avg_acc_df, args)

    vis_avg_acc(best_avg_acc_df, random_avg_acc_df, worst_avg_acc_df, args)
    vis_complex_acc(best_avg_acc_df, random_avg_acc_df, worst_avg_acc_df, args)
    for f in range(2, args.max_feature_space_size):
        vis_complex_acc_for_feat(best_avg_acc_df, random_avg_acc_df, worst_avg_acc_df, f, args)


    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)