import subprocess
import argparse
import pandas as pd
from rich.progress import track
from dask.distributed import LocalCluster, Client

from canonical_task_generation_sim_exp.lib.arguments import parser
from canonical_task_generation_sim_exp.lib.action_space_range import complex_action_space_range
from canonical_task_generation_sim_exp import simulate_user_demos
from canonical_task_generation_sim_exp.generate_canonical_task_archive import create_search_space_task_archive, find_score_spanning_canonical_task_archive
from canonical_task_generation_sim_exp.generate_canonical_task_archive import save_tasks as save_canonical_tasks
from canonical_task_generation_sim_exp.generate_canonical_task_archive import load_score_span_tasks as load_canonical_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import create_complex_task_archive, find_n_hardest_complex_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import save_tasks as save_complex_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import load_tasks as load_complex_tasks
from canonical_task_generation_sim_exp.sample_users import save_search_users, load_search_users, save_test_users, load_test_users, create_user_archive
from canonical_task_generation_sim_exp.learn_reward_function import train, save_learned_weights, load_learned_weights
from canonical_task_generation_sim_exp.evaluate_results import eval, save_eval_results, load_eval_results, avg_complex_task_acc, save_processed_results, load_processed_results
from canonical_task_generation_sim_exp.vis_results import vis_score_v_acc


def main(args):

    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)

    if args.load_search_users:
        search_users_archive = load_search_users(args)
    else:
        search_users_archive = create_user_archive(feat_space_range=(2, args.max_feature_space_size), num_users=args.weight_samples, weight_space=args.weight_space)
        save_search_users(search_users_archive, args)

    if args.load_canonical_tasks:
        print("Loading task search space")
        canonical_task_search_space = load_canonical_tasks("search_space", args)
    else:
        canonical_task_search_space = create_search_space_task_archive(action_space_range=(2, args.max_canonical_action_space_size),
                                                                       feat_space_range=(2, args.max_feature_space_size),
                                                                       num_sampled_tasks=args.num_experiments)
        save_canonical_tasks("search_space", canonical_task_search_space, args)

    if args.load_search_results:
        canonical_tasks_archive = load_canonical_tasks("score_spanning", args)
    else:
        canonical_tasks_archive = find_score_spanning_canonical_task_archive(dask_client=client,
                                                                        user_archive=search_users_archive,
                                                                        task_archive=canonical_task_search_space,
                                                                        action_space_range=(2, args.max_canonical_action_space_size),
                                                                        feat_space_range=(2, args.max_feature_space_size),
                                                                        metric=args.metric,
                                                                        num_sampled_tasks=args.num_experiments,
                                                                        num_sampled_agents=args.weight_samples,
                                                                        max_experiment_len=args.max_experiment_len,
                                                                        num_spanning_tasks=args.num_canonical_tasks,
                                                                        args=args)

        save_canonical_tasks("score_spanning", canonical_tasks_archive, args)

    if not (args.load_results or args.load_predictions):
        if args.load_complex_tasks:
            complex_tasks_archive = load_complex_tasks(args)
        else:
            if args.hardest_complex_tasks:
                complex_tasks_archive = find_n_hardest_complex_tasks(dask_client=client,
                                                                    user_archive=search_users_archive,
                                                                    action_space_range=(args.max_canonical_action_space_size, args.max_complex_action_space_size),
                                                                    feat_space_range=(2, args.max_feature_space_size),
                                                                    num_tasks_per_quadrant=args.num_test_tasks,
                                                                    metric=args.metric,
                                                                    num_sampled_tasks=args.num_experiments,
                                                                    num_sampled_agents=args.weight_samples,
                                                                    max_experiment_len=args.max_experiment_len,
                                                                    args=args)
            else:
                # Actual task sizes now start from max canonical size and go to max complex size
                complex_tasks_archive = create_complex_task_archive(action_space_range=(8, args.max_complex_action_space_size),
                                                                    feat_space_range=(2, args.max_feature_space_size),
                                                                    num_tasks_per_quadrant=args.num_test_tasks)

            save_complex_tasks(complex_tasks_archive, args)

        if args.load_test_users:
            users = load_test_users(args)
        else:
            users = create_user_archive(feat_space_range=(2, args.max_feature_space_size), num_users=args.num_test_users, weight_space=args.weight_space)
            save_test_users(users, args)

        if args.load_user_demos:
            score_spanning_demos_df = simulate_user_demos.load_demos("score_spanning", args)
        else:
            score_spanning_demos_df = None

            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        for complex_as in complex_action_space_range(8, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Simulate user demos - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")
                            demo_df = simulate_user_demos.sim_demos(client, can_task, complex_tasks_archive, feat_users, f, canonical_as, complex_as)
                            demo_df['canonical_task_id'] = can_task_id
                            demo_df.set_index('canonical_task_id', append=True, inplace=True)
                            demo_df.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if score_spanning_demos_df is None:
                                score_spanning_demos_df = demo_df
                            else:
                                score_spanning_demos_df = pd.concat([score_spanning_demos_df, demo_df])

            simulate_user_demos.save_demos("score_spanning", score_spanning_demos_df, args)

        score_spanning_demos_df = score_spanning_demos_df.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])

        if args.load_learned_user_rfs and args.load_user_demos:
            score_spanning_learned_weights_df = load_learned_weights("score_spanning", args)
        elif args.load_learned_user_rfs:
            raise RuntimeError("To load already learned rfs, user demos must also be loaded")
        else:
            score_spanning_learned_weights_df = None
            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    demo_as_df = score_spanning_demos_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        demo = demo_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        for complex_as in complex_action_space_range(8, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Learn reward function - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                            learned_weights = train(client, can_task, complex_tasks_archive, demo, f, canonical_as, complex_as)
                            learned_weights['canonical_task_id'] = can_task_id
                            learned_weights.set_index('canonical_task_id', append=True, inplace=True)
                            learned_weights.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if score_spanning_learned_weights_df is None:
                                score_spanning_learned_weights_df = learned_weights
                            else:
                                score_spanning_learned_weights_df = pd.concat([score_spanning_learned_weights_df, learned_weights])

            save_learned_weights("score_spanning", score_spanning_learned_weights_df, args)

    if not args.load_results:
        if args.load_predictions:
            score_spanning_task_acc_df = load_eval_results("score_spanning", args)
        else:
            score_spanning_task_acc_df = None
            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    can_as_task_df = canonical_tasks_archive.xs((f, canonical_as), level=["feat_dim", "num_actions"], drop_level=False)
                    idx_vals = can_as_task_df.index.get_level_values(level="id")
                    demo_as_df = score_spanning_demos_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    weights_as_df = score_spanning_learned_weights_df.xs((f, canonical_as), level=["feat_dim", "num_canonical_actions"], drop_level=False)
                    for can_task_id in idx_vals:
                        can_task = can_as_task_df.xs((can_task_id,), level=["id"])
                        demo = demo_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        weights = weights_as_df.xs((can_task_id,), level=["canonical_task_id"])
                        for complex_as in complex_action_space_range(8, args.max_complex_action_space_size):

                            print("---------------------------------")
                            print(f"Learn reward function - Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                            task_acc = eval(client, complex_tasks_archive, weights, demo, f, canonical_as, complex_as)
                            task_acc['canonical_task_id'] = can_task_id
                            task_acc.set_index('canonical_task_id', append=True, inplace=True)
                            task_acc.reorder_levels(["feat_dim", "num_canonical_actions", "canonical_task_id", "num_complex_actions", "complex_task_id", "uid"])
                            if score_spanning_task_acc_df is None:
                                score_spanning_task_acc_df = task_acc
                            else:
                                score_spanning_task_acc_df = pd.concat([score_spanning_task_acc_df, task_acc])

            save_eval_results("score_spanning", score_spanning_task_acc_df, args)
            print(score_spanning_task_acc_df)

    vis_score_v_acc(score_spanning_task_acc_df, canonical_tasks_archive, args)



    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)