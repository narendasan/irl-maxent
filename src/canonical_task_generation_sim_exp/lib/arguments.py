import argparse

parser = argparse.ArgumentParser(description='Canonical task generation experiments')
parser.add_argument('--num-experiments', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--weight-samples', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--max-canonical-action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions for canonical tasks')
parser.add_argument('--max-complex-action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions complex tasks')
parser.add_argument('--max-feature-space-size', type=int,
                    default=4,
                    help='Dimensionality of feature space describing actions')
parser.add_argument('--max-experiment-len', type=int,
                    default=100,
                    help='Maximum number of steps taken in each experiment')
parser.add_argument("--verbose", action='store_true', help='Print selected tasks')

parser.add_argument("--load-canonical-tasks", action='store_true', help="Load canonical tasks already generated using the provided settings")
parser.add_argument("--load-complex-tasks", action='store_true', help="Load complex tasks already generated using the provided settings")
parser.add_argument("--load-test-users", action='store_true', help="Load test users using the provided settings")
parser.add_argument("--load-user-demos", action='store_true', help="Load already captured user demos using the provided settings")
parser.add_argument("--load-learned-user-rfs", action='store_true', help="Load already learned user reward functions using the provided settings (requires load-user-demos)")
parser.add_argument("--load-predictions", action='store_true', help="Load a task from a saved file")
parser.add_argument("--load-results", action='store_true', help="Load a task from a saved file")
parser.add_argument("--load-version", type=str, default=None, help="Load a task from a saved file")
parser.add_argument("--version", type=str, default=None, help="Load a task from a saved file")


parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes to run experiements")
parser.add_argument("--metric", type=str, default="unique-trajectories",
                    help="What metric to use to determine if a task is good a distingushing reward functions")
parser.add_argument("--weight-space", type=str, default="normal", help="What space to sample weights from")
parser.add_argument("--iterations", type=int, default=1, help="What iteration of the experiment")
parser.add_argument("--headless", action='store_true', help='Dont show figures')
parser.add_argument("--only-vis", action='store_true', help='Skip the experiments and just analyize results')
parser.add_argument("--load-canonical-task-archives", action='store_true', help='Load already generated canonical tasks')
parser.add_argument("--num-test-users", type=int, default=10, help="Number of users to test on")
parser.add_argument("--num-test-tasks", type=int, default=10, help="Number of complex tasks to sample for each feat, action pair")
parser.add_argument("--num-canonical-tasks", type=int, default=10, help="Number of canonical tasks used to span the space of dispersion scores")


def args_to_prefix(args, load: bool = False):
    path = f"num_exp{args.num_experiments}-weight_samples{args.weight_samples}-max_canonical_action_space_size{args.max_canonical_action_space_size}-max_complex_action_space_size{args.max_complex_action_space_size}-max_feat_size{args.max_feature_space_size}-max_exp_len{args.max_experiment_len}-metric_{args.metric}-weight_space_{args.weight_space}"
    if load == False and args.load_version is not None:
        return f"{path}_{args.load_version}"
    elif args.version is not None:
        return f"{path}_{args.version}"
    else:
        return path


def out_path(args, kind: str = "results", owner: str = None, load: bool = False):
    import pathlib
    p = pathlib.Path(f"{kind}/{owner}/{args_to_prefix(args, load)}/")
    p.mkdir(parents=True, exist_ok=True)
    return p