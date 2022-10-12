import argparse
from typing import List, Tuple
from task import RIRLTask
from agent import GreedyAgent, VIAgent, to_key
import numpy as np
from dataclasses import dataclass
from functools import reduce
import math

from rich.progress import track

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

args = parser.parse_args()

@dataclass
class TrajectoryResult:
    trajectory: List[Tuple]
    num_ties: int

@dataclass
class TaskWithMetric:
    task: RIRLTask
    metric: float

def run_experiment(task, agent):
    current_state = np.zeros((task.num_actions), dtype=np.uint8)
    end_state = np.ones((task.num_actions), dtype=np.uint8)
    step = 0
    num_trajectory_ties = 0
    trajectory = []
    while not np.equal(current_state, end_state).all() and step < args.max_experiment_len:
        #print(f"Current state: {current_state}")
        action, num_ties = agent.act(current_state) # NOTE: NOT SURE IF THIS MAKES SENSE, BASICALLY REPORT BACK HOW MANY AMBIGUOUS STATES THERE ARE WITH THESE WEIGHTS
        _, next_state = RIRLTask.transition(current_state, action)
        trajectory.append((current_state, action))
        current_state = next_state
        step += 1
        num_trajectory_ties += num_ties

    trajectory.append((current_state, None))
    return TrajectoryResult(trajectory, num_trajectory_ties)

def V(action_space_size, r_max):
    # Calculates V as an n-d cube where features all have value 1
    # i.e. R_max = sum(feat)

    # Could also do a tighter?? bound by using a sphere.
    return (2 * r_max) ** action_space_size

def metric(trajectories):
    def tie_sum(a, b):
        return int(a.num_ties + b.num_ties)

    return reduce(tie_sum, trajectories)

def trajectory_to_string(t: TrajectoryResult) -> str:
    t_str = str([hex(to_key(s)) for s,_ in t.trajectory])
    return t_str

def main():
    #TODO: Sample at the start a bunch of agent weights (~1000) [1xnum_feats]
    agent_feature_weights = np.random.normal(loc=0.0, scale=1.0, size=(args.weight_samples, args.feature_space_size))

    for action_space_size in range(2, args.max_action_space_size + 1):
        experiments = {}
        min_ties = math.inf
        for t in track(range(args.num_experiments), description=f"Sampling envs {args.num_experiments} with action space size {action_space_size} and testing with {args.weight_samples} pre-sampled agents"):
            feats = np.random.random((action_space_size, args.feature_space_size))
            task = RIRLTask(features=feats)

            agents = []
            for w in agent_feature_weights:
                agents.append(VIAgent(task, feat_weights=w))

            trajectories = []
            for a in agents:
                trajectory = run_experiment(task, a)
                trajectories.append(trajectory_to_string(trajectory))

            #TODO: For each task, count the number of unqiue trajectories when running against the 1000 agents.
            # Save the most unqiue and least unqiue tasks
            unique_trajectories = set(trajectories)
            if len(unique_trajectories) not in experiments:
                experiments[len(unique_trajectories)] = [task]
            else:
                experiments[len(unique_trajectories)].append(task)

        max_val = max(list(experiments.keys()))
        print(f"{len(experiments[max_val])} Tasks with the most unique trajectories ({max_val})")
        if args.verbose:
            print(experiments[max_val])

if __name__ == "__main__":
    print(args)
    main()
