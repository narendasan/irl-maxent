import argparse
from typing import List, Tuple, Dict
from task import RIRLTask
from agent import GreedyAgent, VIAgent
import numpy as np
from dataclasses import dataclass
from functools import reduce
from collections import namedtuple
import math

import pickle as pkl

from rich.progress import track
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy

from dask.distributed import Client, LocalCluster

sns.set(rc={"figure.figsize": (20, 10)})

parser = argparse.ArgumentParser(description='Repeated IRL experiements')
parser.add_argument('--num-experiments', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--weight-samples', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--max-action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions')
parser.add_argument('--feature-space-size', type=int,
                    default=3,
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
parser.add_argument("--iteration", type=int, default=0, help="What iteration of the experiment")
parser.add_argument("--headless", action='store_true', help='Dont show figures')

args = parser.parse_args()


@dataclass
class TrajectoryResult:
    trajectory: List[Tuple]
    num_ties: int
    cumulative_seen_features: np.array


def run_experiment(task_features, task_preconditions, agent_weights):
    task = RIRLTask(features=task_features, preconditions=task_preconditions)
    agent = VIAgent(task, feat_weights=agent_weights)

    current_state = np.zeros((task.num_actions), dtype=np.uint8)
    end_state = np.ones((task.num_actions), dtype=np.uint8)
    step = 0
    num_trajectory_ties = 0
    trajectory = []
    while not np.equal(current_state, end_state).all() and step < args.max_experiment_len:
        # print(f"Current state: {current_state}")
        action, num_ties = agent.act(
            current_state)  # NOTE: NOT SURE IF THIS MAKES SENSE, BASICALLY REPORT BACK HOW MANY AMBIGUOUS STATES THERE ARE WITH THESE WEIGHTS
        _, next_state = task.transition(current_state, action)
        trajectory.append((current_state, action))
        current_state = next_state
        step += 1
        num_trajectory_ties += num_ties

    trajectory.append((current_state, None))
    return TrajectoryResult(trajectory, num_trajectory_ties, agent.cumulative_seen_state_features)


def unique_trajectories_metric(experiements: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    def trajectory_to_string(t: TrajectoryResult) -> str:
        t_str = str([hex(RIRLTask.state_to_key(s)) for s, _ in t.trajectory])
        return t_str

    task_scores = {}
    for i, trajectories in experiements.items():
        trajectory_strings = [trajectory_to_string(t) for t in trajectories]
        unique_trajectories = set(trajectory_strings)
        task_scores[i] = len(unique_trajectories) / len(trajectories)
    return task_scores


def unique_cumulative_features_metric(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    task_scores = {}
    for i, trajectories in experiments.items():
        cumulative_feats = [t.cumulative_seen_features for t in trajectories]
        unique_cumulative_feats = np.unique(np.vstack(cumulative_feats), axis=0)
        task_scores[i] = unique_cumulative_feats.shape[0] / len(cumulative_feats)
    return task_scores


# TODO: Implement a dispersion metric that uses cosine similarity instead of distance. See notes for details
# TODO: Figure out why the plot shows some particularly bad tasks for the best and good for the worst.
def dispersion_metric(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    def F(trajectory_i: TrajectoryResult):
        return trajectory_i.cumulative_seen_features / len(trajectory_i.trajectory)

    task_scores = {}
    for i, trajectories in experiments.items():
        F_is = np.vstack([F(t) for t in trajectories])
        F_bar = np.mean(F_is, axis=0)
        tiled_F_bar = np.tile(F_bar, (len(trajectories), 1))
        diff = F_is - tiled_F_bar
        dispersion_inner = np.einsum("ij,ij -> i", diff, diff)
        assert (dispersion_inner.shape == (len(trajectories),))
        task_scores[i] = np.sum(dispersion_inner) / (len(trajectories) - 1)
    return task_scores


def normed_dispersion_metric(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    def F(trajectory_i: TrajectoryResult):
        return trajectory_i.cumulative_seen_features / len(trajectory_i.trajectory)

    def H(f_trajectory_i: np.array):
        f_trajectory_i_hat = f_trajectory_i / np.linalg.norm(f_trajectory_i)
        return f_trajectory_i_hat

    task_scores = {}
    for i, trajectories in experiments.items():
        F_is = np.vstack([H(F(t)) for t in trajectories])
        F_bar = np.mean(F_is, axis=0)
        tiled_F_bar = np.tile(F_bar, (len(trajectories), 1))
        diff = F_is - tiled_F_bar
        dispersion_inner = np.einsum("ij,ij -> i", diff, diff)
        assert (dispersion_inner.shape == (len(trajectories),))
        task_scores[i] = np.sum(dispersion_inner) / (len(trajectories) - 1)
    return task_scores


def cos_dispersion_metric(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    def F(trajectory_i: TrajectoryResult):
        return trajectory_i.cumulative_seen_features / len(trajectory_i.trajectory)

    def H(f_trajectory_i: np.array):
        f_trajectory_i_hat = f_trajectory_i / np.linalg.norm(f_trajectory_i)
        assert (f_trajectory_i_hat.shape[-1] == 3)
        hxy = np.hypot(f_trajectory_i_hat[0], f_trajectory_i_hat[1])
        r = np.hypot(hxy, f_trajectory_i_hat[2])
        el = np.arctan2(f_trajectory_i_hat[2], hxy)
        az = np.arctan2(f_trajectory_i_hat[1], f_trajectory_i_hat[0])
        # only angles?
        return np.hstack((el, az))

    task_scores = {}
    for i, trajectories in experiments.items():
        F_is = np.vstack([H(F(t)) for t in trajectories])
        F_bar = np.mean(F_is, axis=0)
        tiled_F_bar = np.tile(F_bar, (len(trajectories), 1))
        diff = F_is - tiled_F_bar
        dispersion_inner = np.einsum("ij,ij -> i", diff, diff)
        assert (dispersion_inner.shape == (len(trajectories),))
        task_scores[i] = np.sum(dispersion_inner) / (len(trajectories) - 1)
    return task_scores


Metric = namedtuple("Metric", ["name", "func"])

METRICS = {
    "unique-trajectories": Metric("unique trajectories / sampled weights", unique_trajectories_metric),
    "unique-cumulative-features": Metric("unique cumulative features / sampled weights",
                                         unique_cumulative_features_metric),
    "dispersion": Metric("dispersion", dispersion_metric),
    "normed-dispersion": Metric("normed-dispersion", cos_dispersion_metric),
    "cos-dispersion": Metric("cos-dispersion", cos_dispersion_metric),
}


def task_feat_subset(task_feats: Dict[int, np.array],
                     task_trans: Dict[int, np.array],
                     task_ids: List[int]) -> List[np.array]:
    return [[task_feats[id], task_trans[id]] for id in task_ids]


def task_subset(task_feats: Dict[int, np.array],
                task_trans: Dict[int, np.array],
                task_ids: List[int]) -> List[RIRLTask]:
    return [RIRLTask(features=f, preconditions=p) for f, p in task_feat_subset(task_feats, task_trans, task_ids)]


def sample_halton(shape: Tuple) -> np.array:
    rng = scipy.stats.qmc.Halton(d=shape[1], scramble=False)
    return rng.random(n=shape[0] + 1)[1:]  # Skip the first one which is always 0,0,0 when scramble is off


def sample_spherical(shape: Tuple) -> np.array:
    phi = np.linspace(0, np.pi, 2000)
    theta = np.linspace(0, 2 * np.pi, 4000)
    x = 0.5 * np.outer(np.sin(theta), np.cos(phi)) + 0.5
    y = 0.5 * np.outer(np.sin(theta), np.sin(phi)) + 0.5
    z = 0.5 * np.outer(np.cos(theta), np.ones_like(phi)) + 0.5

    x = x.flatten()
    x = x[np.newaxis, :]
    x = x.T

    y = y.flatten()
    y = y[np.newaxis, :]
    y = y.T

    z = z.flatten()
    z = z[np.newaxis, :]
    z = z.T

    space = np.hstack((x, y, z))
    sample = space[np.random.choice(np.arange(space.shape[0]), shape[0], replace=False), :]

    return sample


WEIGHT_SPACE = {
    "normal": np.random.normal(loc=0.0, scale=1.0, size=(args.weight_samples, args.feature_space_size)),
    "halton": sample_halton(shape=(args.weight_samples, args.feature_space_size)),
    "spherical": sample_spherical(shape=(args.weight_samples, args.feature_space_size))
}


def recursive_dependencies(actions, dependencies, result=None):
    """
    Recursively populate all dependencies of actions.
    """

    if result is None:
        result = set()
    for a in actions:
        result.update(dependencies[a])
        recursive_dependencies(dependencies[a], dependencies, result)

    return result


def checkHTN(precondition, all_preconditions):
    """
    Function to check if new action affects the preconditions of other actions.
    Args:
        precondition: preconditions of new action
        all_preconditions: preconditions of all actions in the task

    Returns:
        True if new action can be added to the task without affecting other actions.
    """

    # dependency for each action
    dependencies = [set(np.where(oa_dependency)[0]) for oa_dependency in all_preconditions]

    # dependency for new action
    a_dependency = set(np.where(precondition)[0])
    num_dependencies = len(a_dependency)

    if num_dependencies > 0:

        verification = True

        # check if new action affects the dependency of previous actions
        for oa, oa_dependency in enumerate(dependencies):
            if oa not in a_dependency:
                common_dependencies = a_dependency.intersection(oa_dependency)
                unique_dependencies = a_dependency.symmetric_difference(oa_dependency)
                if common_dependencies and unique_dependencies:
                    # check if the common dependencies are dependent on the unique
                    all_oa_dependencies = recursive_dependencies(common_dependencies, dependencies)
                    if all_oa_dependencies != unique_dependencies:
                        verification = False
                        break
    else:
        # no check required for actions without any dependencies
        verification = True

    return verification


def generate_task(num_actions, num_features, feature_space=None):

    # TODO: directly take feature space as input
    if not feature_space:
        if num_actions < 3:
            feature_bounds = [(0, num_actions),  # which part
                              (0, num_actions)]  # which tool
        else:
            feature_bounds = [(0, math.ceil(num_actions/2)),
                              (0, math.ceil(num_actions/2))]

        feature_space = []
        for lb, ub in feature_bounds:
            feature_space.append([f_val for f_val in range(lb, ub, 1)])

    task_actions, task_preconditions = [], []
    for i in range(num_actions):
        new_action = []
        for j in range(num_features):
            if j < len(feature_space):
                new_action.append(np.random.choice(feature_space[j]))
            else:
                new_action.append(np.random.random())

        precondition_verified = False
        while not precondition_verified:
            action_precondition = np.zeros(num_actions)
            for oa in range(len(task_actions)):
                dep = np.random.choice([0, 1], p=[0.60, 0.40])
                if dep == 1:
                    action_precondition[oa] = 1

            precondition_verified = checkHTN(action_precondition, task_preconditions)

        task_actions.append(new_action)
        task_preconditions.append(action_precondition)

    return task_actions, task_preconditions


def main():
    try:
        assert (args.metric in list(METRICS.keys()))
    except:
        raise RuntimeError(f"Invalid metric {args.metric} (valid metrics: {list(METRICS.keys())})")

    try:
        assert (args.weight_space in list(WEIGHT_SPACE.keys()))
    except:
        raise RuntimeError(f"Invalid weight space {args.weight_space} (valid weight spaces: {list(WEIGHT_SPACE.keys())})")
   # Sample at the start a bunch of agent weights (~1000) [1xnum_feats]
    # TODO: Look at other sampling methods to more effectively cover the trajectory space.
    # TODO: 12/14: Use DDP to sample them
    # TODO: 1/11: Fix Spherical

    agent_feature_weights = WEIGHT_SPACE[args.weight_space]
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = 0.5 * np.outer(np.sin(theta), np.cos(phi)) + 0.5
    print(x.shape)
    y = 0.5 * np.outer(np.sin(theta), np.sin(phi)) + 0.5
    z = 0.5 * np.outer(np.cos(theta), np.ones_like(phi)) + 0.5
    as_x, as_y, as_z = agent_feature_weights[:, 0], agent_feature_weights[:, 1], agent_feature_weights[:, 2]
    plt.figure(figsize=(11, 8))
    plot_axes = plt.axes(projection='3d')
    plot_axes.scatter3D(as_x, as_y, as_z)
    plot_axes.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    plt.savefig(
        f"figures/sampled_agents_distribution_{args.weight_samples}_feat_space_size_{args.feature_space_size}_sampled_tasks{args.num_experiments}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.png")

    if not args.headless:
        plt.show()
        plt.close()

    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )
    client = Client(cluster)

    if args.load_from_file:
        # TODO: Use the simulate_users code from bayesian-learning and replace the Cannonical features with the best and worst task features to simulate users (cannonical_demos.csv)
        # Then go to experiment sim (run_maxent=True, map_estimate=True, all else false, set training_samples = 30-50, test_samples=1)
        # Plot the accuracy for the best vs worst result of experiment sim (bar chart)
        task_list = np.load(args.load_from_file, allow_pickle=True)
        print(task_list)
        for i, task in enumerate(task_list):
            agents = []
            for w in agent_feature_weights:
                agents.append(VIAgent(task, feat_weights=w))

            trajectories = []
            for a in agents:
                trajectory = run_experiment(task, a)
                trajectories.append(trajectory)

            with open(f"trajectories_for_{args.load_from_file}_{i}.pkl", "wb") as f:
                pkl.dump(trajectories, f)

    else:
        experiement_results_by_action_space = {}

        best_task_ids_by_action_space = {}
        random_task_ids_by_action_space = {}
        worst_task_ids_by_action_space = {}

        best_score_by_action_space = {}
        random_score_by_action_space = {}
        worst_score_by_action_space = {}

        best_task_feats_by_action_space = {}
        random_task_feats_by_action_space = {}
        worst_task_feats_by_action_space = {}

        best_task_trans_by_action_space = {}
        random_task_trans_by_action_space = {}
        worst_task_trans_by_action_space = {}

        for action_space_size in range(2, args.max_action_space_size + 1):
            task_feats, task_transitions = {}, {}
            for i in range(args.num_experiments):
                task_feats[i], task_transitions[i] = generate_task(action_space_size, args.feature_space_size)
            experiments = {}
            min_ties = math.inf

            for i in track(range(args.num_experiments),
                           description=f"Sampling envs {args.num_experiments} with action space size {action_space_size} and testing with {args.weight_samples} pre-sampled agents"):
                trajectories = []
                # for a in agents:
                #    trajectory = run_experiment(task, a)
                # TODO: Replace trajectory to string to summed feature values over the trajectories
                #    trajectories.append(trajectory_to_string(trajectory))
                futures = client.map(lambda e: run_experiment(e[0], e[1], e[2]),
                                     list(zip([task_feats[i]] * len(agent_feature_weights),
                                              [task_transitions[i]] * len(agent_feature_weights),
                                              agent_feature_weights)))
                trajectory_results = client.gather(futures)
                experiments[i] = trajectory_results

            scores_for_tasks = METRICS[args.metric].func(experiments)

            max_score = max(scores_for_tasks.values())
            min_score = min(scores_for_tasks.values())

            best_tasks = [t_id for t_id, score in scores_for_tasks.items() if score == max_score]
            random_tasks = np.random.choice(list(scores_for_tasks.keys()), 1)
            random_score = np.average([scores_for_tasks[t] for t in random_tasks])
            worst_tasks = [t_id for t_id, score in scores_for_tasks.items() if score == min_score]

            # Save best and worst tasks (number of unique trajectories) to a file
            print(f"{len(best_tasks)} Tasks with best {METRICS[args.metric].name} ({max_score})")
            print(f"{len(random_tasks)} Tasks with random {METRICS[args.metric].name} (avg: {random_score})")
            print(f"{len(worst_tasks)} Tasks with worst {METRICS[args.metric].name} ({min_score})")
            if args.verbose:
                print(f"Best tasks: {task_subset(task_feats, task_transitions, best_tasks)}")
                print(f"Random tasks: {task_subset(task_feats, task_transitions, best_tasks)}")
                print(f"Worst tasks: {task_subset(task_feats, task_transitions, worst_tasks)}")

            np.save(
                f"results/best_actions{args.max_action_space_size}_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}",
                task_subset(task_feats, task_transitions, best_tasks))
            np.save(
                f"results/random_actions{args.max_action_space_size}_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}",
                task_subset(task_feats, task_transitions, random_tasks))
            np.save(
                f"results/worst_actions{args.max_action_space_size}_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}",
                task_subset(task_feats, task_transitions, worst_tasks))

            experiement_results_by_action_space[action_space_size] = experiments

            best_task_ids_by_action_space[action_space_size] = best_tasks
            random_task_ids_by_action_space[action_space_size] = random_tasks
            worst_task_ids_by_action_space[action_space_size] = worst_tasks

            best_score_by_action_space[action_space_size] = max_score
            random_score_by_action_space[action_space_size] = random_score
            worst_score_by_action_space[action_space_size] = min_score

            best_task_feats_by_action_space[action_space_size] = [task_feats[k] for k in best_tasks]
            random_task_feats_by_action_space[action_space_size] = [task_feats[k] for k in random_tasks]
            worst_task_feats_by_action_space[action_space_size] = [task_feats[k] for k in worst_tasks]

            best_task_trans_by_action_space[action_space_size] = [task_transitions[k] for k in best_tasks]
            random_task_trans_by_action_space[action_space_size] = [task_transitions[k] for k in random_tasks]
            worst_task_trans_by_action_space[action_space_size] = [task_transitions[k] for k in worst_tasks]

        # TODO: Save the actual metric numbers to a table

        print(
            f"Action space vs. Number of unqiue trajectories for {args.weight_samples} sampled agents based on {METRICS[args.metric].name}: {best_score_by_action_space}")
        print(
            f"Action space vs. Number of unqiue trajectories for {args.weight_samples} sampled agents based on {METRICS[args.metric].name}: {random_score_by_action_space}")
        print(
            f"Action space vs. Number of unqiue trajectories for {args.weight_samples} sampled agents based on {METRICS[args.metric].name}: {worst_score_by_action_space}")

        metric_df = pd.DataFrame({
            "Action Space Size": list(best_score_by_action_space.keys()),
            f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Best Task for Action Space Size N": list(
                best_score_by_action_space.values()),
            f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Random Task for Action Space Size N": list(
                random_score_by_action_space.values()),
            f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})\nfor Worst Task for Action Space Size N": list(
                worst_score_by_action_space.values())
        })
        metric_df.name = f"Action Space Size vs. Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})"

        plt.figure()
        plot = sns.lineplot(x="Action Space Size",
                            y=f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})",
                            hue="Task Class",
                            data=pd.melt(metric_df,
                                         ["Action Space Size"],
                                         var_name="Task Class",
                                         value_name=f"Reward Function Uniqueness Metric ({METRICS[args.metric].name}/{args.weight_space})"))
        plot.set(
            title=f"Action space vs. distingushable reward function metric ({METRICS[args.metric].name}) for {args.weight_samples} sampled agents (feature space size={args.feature_space_size}, sampled tasks={args.num_experiments})")
        plt.savefig(
            f"figures/action_space_vs_metric_sampled_agents_{args.weight_samples}_feat_space_size_{args.feature_space_size}_sampled_tasks{args.num_experiments}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.png")

        if not args.headless:
            plt.show()

        with open(
                f"results/best_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(best_task_feats_by_action_space, f)
        with open(
                f"results/random_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(random_task_feats_by_action_space, f)
        with open(
                f"results/worst_exp{args.num_experiments}_feat{args.feature_space_size}_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(worst_task_feats_by_action_space, f)

        with open(
                f"results/best_exp{args.num_experiments}_trans_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(best_task_trans_by_action_space, f)
        with open(
                f"results/random_exp{args.num_experiments}_trans_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(random_task_trans_by_action_space, f)
        with open(
                f"results/worst_exp{args.num_experiments}_trans_metric_{args.metric}_space_{args.weight_space}_{args.iteration}.pkl",
                "wb") as f:
            pkl.dump(worst_task_trans_by_action_space, f)

if __name__ == "__main__":
    print(args)
    main()
