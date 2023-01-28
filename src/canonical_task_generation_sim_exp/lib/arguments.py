import argparse

parser = argparse.ArgumentParser(description='Canonical task generation experiments')
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
parser.add_argument("--only-vis", action='store_true', help='Skip the experiments and just analyize results')

def args_to_prefix(args):
    return f"num_exp{args.num_experiments}-weight_samples{args.weight_samples}-max_action_space_size{args.max_action_space_size}-max_feat_size{args.max_feature_space_size}-max_exp_len{args.max_experiment_len}-metric_{args.metric}-weight_space_{args.weight_space}"

def out_path(args, kind: str = "results", owner: str = None):
    import pathlib
    p = pathlib.Path(f"{kind}/{owner}/{args_to_prefix(args)}/")
    p.mkdir(parents=True, exist_ok=True)
    return p