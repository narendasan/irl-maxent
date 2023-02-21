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
from canonical_task_generation_sim_exp.generate_canonical_task_archive import load_tasks as load_canonical_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import create_complex_task_archive
from canonical_task_generation_sim_exp.generate_complex_task_archive import save_tasks as save_complex_tasks
from canonical_task_generation_sim_exp.generate_complex_task_archive import load_tasks as load_complex_tasks
from canonical_task_generation_sim_exp.sample_users import save_users, load_users, create_user_archive
from canonical_task_generation_sim_exp.learn_reward_function import train, save_learned_weights, load_learned_weights
from canonical_task_generation_sim_exp.evaluate_results import eval, save_eval_results, load_eval_results, avg_complex_task_acc, save_processed_results, load_processed_results
from canonical_task_generation_sim_exp.vis_results import vis_avg_acc, vis_complex_acc, vis_complex_acc_for_feat

def main(args):

    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)


    if not (args.load_results or args.load_predictions):
        if args.load_canonical_tasks:
            best_canonical_tasks = load_canonical_tasks("best", args)
            random_canonical_tasks = load_canonical_tasks("random", args)
            worst_canonical_tasks= load_canonical_tasks("worst", args)
        else:
            best_canonical_tasks, random_canonical_tasks, worst_canonical_tasks = create_canonical_task_archive(dask_client=client,
                                                                                                                action_space_range=(2, args.max_canonical_action_space_size),
                                                                                                                feat_space_range=(2, args.max_feature_space_size),
                                                                                                                weight_space=args.weight_space,
                                                                                                                metric=args.metric,
                                                                                                                num_sampled_tasks=args.num_experiments,
                                                                                                                num_sampled_agents=args.weight_samples,
                                                                                                                max_experiment_len=args.max_experiment_len)

            save_canonical_tasks("best", best_canonical_tasks, args)
            save_canonical_tasks("random", random_canonical_tasks, args)
            save_canonical_tasks("worst", worst_canonical_tasks, args)

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
            print(users)
        else:
            users = create_user_archive(feat_space_range=(2, args.max_feature_space_size), num_users=args.num_test_users, weight_space=args.weight_space)

            save_users(users, args)

        if args.load_user_demos:
            best_demos_df = simulate_user_demos.load_demos("best", args)
            random_demos_df = simulate_user_demos.load_demos("random", args)
            worst_demos_df = simulate_user_demos.load_demos("worst", args)
        else:
            best_demos_df, random_demos_df, worst_demos_df = None, None, None

            for f in range(2, args.max_feature_space_size + 1):
                feat_user_df = users.loc[[f]]
                feat_users = feat_user_df["users"]
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):
                        print("---------------------------------")
                        print(f"Simulate user demos - Kind: best, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")
                        best_demo_df = simulate_user_demos.sim_demos(client, best_canonical_tasks, complex_tasks_archive, feat_users, f, canonical_as, complex_as)
                        if best_demos_df is None:
                            best_demos_df = best_demo_df
                        else:
                            best_demos_df = pd.concat([best_demos_df, best_demo_df])
                        print("---------------------------------")
                        print(f"Simulate user demos - Kind: random, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")
                        random_demo_df = simulate_user_demos.sim_demos(client, random_canonical_tasks, complex_tasks_archive, feat_users, f, canonical_as, complex_as)
                        if random_demos_df is None:
                            random_demos_df = random_demo_df
                        else:
                            random_demos_df = pd.concat([random_demos_df, random_demo_df])
                        print("---------------------------------")
                        print(f"Simulate user demos - Kind: worst, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")
                        worst_demo_df = simulate_user_demos.sim_demos(client, worst_canonical_tasks, complex_tasks_archive, feat_users, f, canonical_as, complex_as)
                        if worst_demos_df is None:
                            worst_demos_df = worst_demo_df
                        else:
                            worst_demos_df = pd.concat([worst_demos_df, worst_demo_df])


            simulate_user_demos.save_demos("best", best_demos_df, args)
            simulate_user_demos.save_demos("random", random_demos_df, args)
            simulate_user_demos.save_demos("worst", worst_demos_df, args)

        if args.load_learned_user_rfs and args.load_user_demos:
            best_learned_weights_df = load_learned_weights("best", args)
            random_learned_weights_df = load_learned_weights("random", args)
            worst_learned_weights_df = load_learned_weights("worst",args)
        elif args.load_learned_user_rfs:
            raise RuntimeError("To load already learned rfs, user demos must also be loaded")
        else:
            best_learned_weights_df, random_learned_weights_df, worst_learned_weights_df = None, None, None
            for f in range(2, args.max_feature_space_size + 1):
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):
                        print("---------------------------------")
                        print(f"Learn reward function - Kind: best, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        best_learned_weights = train(client, best_canonical_tasks, complex_tasks_archive, best_demos_df, f, canonical_as, complex_as)
                        if best_learned_weights_df is None:
                            best_learned_weights_df = best_learned_weights
                        else:
                            best_learned_weights_df = pd.concat([best_learned_weights_df, best_learned_weights])

                        print("---------------------------------")
                        print(f"Learn reward function - Kind: random, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        random_learned_weights = train(client, random_canonical_tasks, complex_tasks_archive, random_demos_df, f, canonical_as, complex_as)
                        if random_learned_weights_df is None:
                            random_learned_weights_df = random_learned_weights
                        else:
                            random_learned_weights_df = pd.concat([random_learned_weights_df, random_learned_weights])

                        print("---------------------------------")
                        print(f"Learn reward function - Kind: worst, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        worst_learned_weights = train(client, worst_canonical_tasks, complex_tasks_archive, worst_demos_df, f, canonical_as, complex_as)
                        if worst_learned_weights_df is None:
                            worst_learned_weights_df = worst_learned_weights
                        else:
                            worst_learned_weights_df = pd.concat([worst_learned_weights_df, worst_learned_weights])

            save_learned_weights("best", best_learned_weights_df, args)
            save_learned_weights("random", random_learned_weights_df, args)
            save_learned_weights("worst", worst_learned_weights_df, args)

    if not args.load_results:
        if args.load_predictions:
            best_complex_task_acc_df = load_eval_results("best", args)
            random_complex_task_acc_df = load_eval_results("random", args)
            worst_complex_task_acc_df = load_eval_results("worst", args)
        else:
            best_complex_task_acc_df, random_complex_task_acc_df, worst_complex_task_acc_df = None, None, None
            for f in range(2, args.max_feature_space_size + 1):
                for canonical_as in range(2, args.max_canonical_action_space_size + 1):
                    for complex_as in complex_action_space_range(args.max_canonical_action_space_size, args.max_complex_action_space_size):
                        print("---------------------------------")
                        print(f"Learn reward function - Kind: best, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        best_task_acc = eval(client, complex_tasks_archive, best_learned_weights_df, best_demos_df, f, canonical_as, complex_as)
                        if best_complex_task_acc_df is None:
                            best_complex_task_acc_df = best_task_acc
                        else:
                            best_complex_task_acc_df = pd.concat([best_complex_task_acc_df, best_task_acc])

                        print("---------------------------------")
                        print(f"Learn reward function - Kind: random, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        random_task_acc = eval(client, complex_tasks_archive, random_learned_weights_df, random_demos_df, f, canonical_as, complex_as)
                        if random_complex_task_acc_df is None:
                            random_complex_task_acc_df = random_task_acc
                        else:
                            random_complex_task_acc_df = pd.concat([random_complex_task_acc_df, random_task_acc])

                        print("---------------------------------")
                        print(f"Learn reward function - Kind: worst, Feat: {f}, Canonical Task Size: {canonical_as}, Complex Task Size: {complex_as}")

                        worst_task_acc = eval(client, complex_tasks_archive, worst_learned_weights_df, worst_demos_df, f, canonical_as, complex_as)
                        if worst_complex_task_acc_df is None:
                            worst_complex_task_acc_df = worst_task_acc
                        else:
                            worst_complex_task_acc_df = pd.concat([worst_complex_task_acc_df, worst_task_acc])

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