from dask.distributed import Client, LocalCluster
import math
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from rich.progress import track
import seaborn as sns

from canonical_task_generation_sim_exp.lib.arguments import parser
from canonical_task_generation_sim_exp.lib.generate_tasks import generate_task
from canonical_task_generation_sim_exp.canonical_task_search.task import RIRLTask
from canonical_task_generation_sim_exp.canonical_task_search.agent import VIAgent
from canonical_task_generation_sim_exp.canonical_task_search.result import TrajectoryResult, TaskFeatsConditions
from canonical_task_generation_sim_exp.canonical_task_search.agent_weights import generate_agent_feature_weights
from canonical_task_generation_sim_exp.canonical_task_search.metrics import score_agent_distingushability, METRICS


sns.set(rc={"figure.figsize": (20, 10)})

def run_experiment(task_features, task_preconditions, agent_weights, max_experiment_len: int = 100):
    task = RIRLTask(features=task_features, preconditions=task_preconditions)
    agent = VIAgent(task, feat_weights=agent_weights)

    current_state = np.zeros((task.num_actions), dtype=np.uint8)
    end_state = np.ones((task.num_actions), dtype=np.uint8)
    step = 0
    num_trajectory_ties = 0
    trajectory = []
    while not np.equal(current_state, end_state).all() and step < max_experiment_len:
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

def task_feat_subset(task_feats: Dict[int, np.array],
                     task_trans: Dict[int, np.array],
                     task_ids: List[int]) -> List[np.array]:
    return [[task_feats[id], task_trans[id]] for id in task_ids]

def task_subset(task_feats: Dict[int, np.array],
                task_trans: Dict[int, np.array],
                task_ids: List[int]) -> List[RIRLTask]:
    return [RIRLTask(features=f, preconditions=p) for f, p in task_feat_subset(task_feats, task_trans, task_ids)]

@dataclass
class SearchResult:
    best: TaskFeatsConditions
    random: TaskFeatsConditions
    worst: TaskFeatsConditions

def find_tasks(dask_client: Client,
              action_space_size: int,
              feat_space_size: int,
              weight_space: str="normal",
              metric: str="dispersion",
              num_sampled_tasks: int = 10,
              num_sampled_agents: int = 10,
              max_experiment_len: int = 100,
              verbose: bool = False) -> SearchResult:

    client = dask_client

    agent_feature_weights = generate_agent_feature_weights(num_sampled_agents, feat_space_size, weight_space)

    task_feats, task_transitions = {}, {}
    for i in range(num_sampled_tasks):
        feats, transitions = generate_task(action_space_size, feat_space_size, precondition_probs=(0.3, 0.7))
        task_feats[i] = feats
        task_transitions[i] = transitions

    experiments = {}
    min_ties = math.inf

    for i in track(range(num_sampled_tasks),
                           description=f"Sampling envs {num_sampled_tasks} with action space size {action_space_size}, feats size {feat_space_size} and testing with {num_sampled_agents} pre-sampled agents"):
        # trajectories = []
        # for a in agents:
        #    trajectory = run_experiment(task, a)
        # TODO: Replace trajectory to string to summed feature values over the trajectories
        #    trajectories.append(trajectory_to_string(trajectory))
        futures = client.map(lambda e: run_experiment(e[0], e[1], e[2], max_experiment_len),
                                list(zip([task_feats[i]] * len(agent_feature_weights),
                                        [task_transitions[i]] * len(agent_feature_weights),
                                        agent_feature_weights)))

        trajectory_results = client.gather(futures)
        experiments[i] = trajectory_results

    scores_for_tasks = score_agent_distingushability(experiments, metric_key=metric)

    max_score = max(scores_for_tasks.values())
    min_score = min(scores_for_tasks.values())

    best_tasks = [t_id for t_id, score in scores_for_tasks.items() if score == max_score] #only take one if there is a tie
    random_tasks = np.random.choice(list(scores_for_tasks.keys()), 1)
    random_score = np.average([scores_for_tasks[t] for t in random_tasks])
    worst_tasks = [t_id for t_id, score in scores_for_tasks.items() if score == min_score]

    # Save best and worst tasks (number of unique trajectories) to a file
    print(f"{len(best_tasks)} Tasks with best {METRICS[metric].name} [action space: {action_space_size}, feat space: {feat_space_size}] ({max_score})")
    print(f"{len(random_tasks)} Tasks with random {METRICS[metric].name}  [action space: {action_space_size}, feat space: {feat_space_size}] (avg: {random_score})")
    print(f"{len(worst_tasks)} Tasks with worst {METRICS[metric].name}  [action space: {action_space_size}, feat space: {feat_space_size}] ({min_score})")

    if verbose:
        print(f"Best tasks: {task_subset(task_feats, task_transitions, best_tasks)}")
        print(f"Random tasks: {task_subset(task_feats, task_transitions, best_tasks)}")
        print(f"Worst tasks: {task_subset(task_feats, task_transitions, worst_tasks)}")

    if len(best_tasks) > 1:
        print(f"There was a tie in the best task, selecting 1 of {len(best_tasks)} tasks as the search result")
    best_task_id = best_tasks[0] #only take one if there is a tie

    random_task_id = np.random.choice(list(scores_for_tasks.keys()), 1)[0]

    if len(worst_tasks) > 1:
        print(f"There was a tie in the worst task, selecting 1 of {len(worst_tasks)} tasks as the search result")
    worst_task_id = worst_tasks[0]

    best_task = TaskFeatsConditions(features=task_feats[best_task_id],
                                    preconditions=task_transitions[best_task_id],
                                    score=scores_for_tasks[best_task_id])

    random_task = TaskFeatsConditions(features=task_feats[random_task_id],
                                      preconditions=task_transitions[random_task_id],
                                      score=scores_for_tasks[random_task_id])

    worst_task = TaskFeatsConditions(features=task_feats[worst_task_id],
                                     preconditions=task_transitions[worst_task_id],
                                     score=scores_for_tasks[worst_task_id])

    return SearchResult(best=best_task, random=random_task, worst=worst_task)

def main(args):
    cluster = LocalCluster(
        processes=True,
        n_workers=args.num_workers,
        threads_per_worker=1
    )

    client = Client(cluster)

    for f in range(3, args.max_feature_space_size + 1):
        for a in range(2, args.max_action_space_size + 1):
            find_tasks(dask_client=client,
                        action_space_size=a,
                        feat_space_size=f,
                        weight_space=args.weight_space,
                        metric=args.metric,
                        num_sampled_tasks=args.num_experiments,
                        num_sampled_agents=args.weight_samples,
                        max_experiment_len=args.max_experiment_len,
                        verbose=True)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)