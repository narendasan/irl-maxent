import argparse
from task import RIRLTask
from agent import GreedyAgent, VIAgent
import numpy as np

parser = argparse.ArgumentParser(description='Repeated IRL experiements')
parser.add_argument('--num-experiments', type=int,
                    default=3,
                    help='Number of experiments to run')
parser.add_argument('--action-space-size', type=int,
                    default=3,
                    help='Number of different possible actions')
parser.add_argument('--feature-space-size', type=int,
                    default=3,
                    help='Dimensionality of feature space describing actions')
parser.add_argument('--max-experiment-len', type=int,
                    default=100,
                    help='Maximum number of steps taken in each experiment')

args = parser.parse_args()

def run_experiment(task, agent):
    current_state = np.zeros((task.num_actions), dtype=np.uint8)
    end_state = np.ones((task.num_actions), dtype=np.uint8)
    step = 0
    trajectory = []
    while not np.equal(current_state, end_state).all() and step < args.max_experiment_len:
        print(f"Current state: {current_state}")
        action = agent.act(current_state)
        _, next_state = RIRLTask.transition(current_state, action)
        trajectory.append((current_state, action))
        current_state = next_state
        step += 1

    trajectory.append((current_state, None))
    return trajectory

def main():
    tasks = []
    agents = []
    for i in range(args.num_experiments):
        feats = np.random.random((args.action_space_size, args.feature_space_size))
        tasks.append(RIRLTask(features=feats))
        agents.append(VIAgent(tasks[-1]))

    trajectories = []
    for t, a in zip(tasks, agents):
        trajectories.append(run_experiment(t, a))
        print(trajectories[-1])




if __name__ == "__main__":
    print(args)
    main()