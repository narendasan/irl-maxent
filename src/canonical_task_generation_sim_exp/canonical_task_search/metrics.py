from typing import List, Dict
import numpy as np
from collections import namedtuple
import math

from canonical_task_generation_sim_exp.canonical_task_search.result import TrajectoryResult
from canonical_task_generation_sim_exp.canonical_task_search.task import RIRLTask

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

def obs_trajectory_dist_volume_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:

    def trajectory_to_string(t: TrajectoryResult) -> str:
        t_str = str([hex(RIRLTask.state_to_key(s)) for s, _ in t.trajectory])
        return t_str

    task_scores = {}
    for i, trajectories in experiments.items():
        trajectory_strings = [trajectory_to_string(t) for t in trajectories]
        trajectory_obs = {}
        for s in trajectory_strings:
            if s in trajectory_obs:
                trajectory_obs[s] += 1
            else:
                trajectory_obs[s] = 1

        trajectory_dist = []
        for _,v in trajectory_obs.items():
            trajectory_dist.append(v / len(trajectory_strings))


        task_scores[i] = np.sum(np.array(trajectory_dist) ** 2)

    return task_scores

def reward_dist_volume_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:

    task_scores = {}
    for i, trajectories in experiments.items():
        trajectory_rewards = [t.cumulative_reward for t in trajectories]
        trajectory_rewards_obs = {}
        for r in trajectory_rewards:
            if r in trajectory_rewards_obs:
                trajectory_rewards_obs[r] += 1
            else:
                trajectory_rewards_obs[r] = 1

        trajectory_dist = []
        for _,v in trajectory_rewards_obs.items():
            trajectory_dist.append(v / len(trajectory_rewards))


        task_scores[i] = np.sum(np.array(trajectory_dist) ** 2)

    return task_scores

def obs_feat_dist_volume_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    def feat_to_string(t: np.array) -> str:
        t_str = str(t)
        return t_str

    task_scores = {}
    for i, trajectories in experiments.items():
        trajectory_feats = [feat_to_string(t.cumulative_seen_features) for t in trajectories]
        trajectory_feats_obs = {}
        for r in trajectory_feats:
            if r in trajectory_feats_obs:
                trajectory_feats_obs[r] += 1
            else:
                trajectory_feats_obs[r] = 1

        trajectory_dist = []
        for _,v in trajectory_feats_obs.items():
            trajectory_dist.append(v / len(trajectory_feats_obs))


        task_scores[i] = np.sum(np.array(trajectory_dist) ** 2)

    return task_scores

def reward_dist_volume_removal_v2(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    task_scores = {}
    for i, demos in experiments.items():
        traj_keys = set()
        probs = []
        for d in demos:
            max_reward = d.cumulative_reward
            max_trajectory_keys = []
            for k, r in d.possible_rewards.items():
                # A bit dangerous perhaps since comparing floats
                if math.isclose(r, max_reward):
                    max_trajectory_keys.append(k)

            w = 1 / len(max_trajectory_keys)

            traj_probs = {}
            for k in d.possible_rewards.keys():
                traj_keys.add(k)
                if k in max_trajectory_keys:
                    traj_probs[k] = w
                else:
                    traj_probs[k] = 0
            probs.append(traj_probs)

        traj_keys = list(traj_keys)
        aligned_probs = []
        for p in probs:
            ap = []
            for k in traj_keys:
                ap.append(p[k])
            aligned_probs.append(ap)

        aligned_probs = np.array(aligned_probs)
        M = len(aligned_probs)
        task_scores[i] = np.sum((np.sum(aligned_probs, axis=0) / M) ** 2)

    return task_scores

def information_gain(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    task_scores = {}
    for i, demos in experiments.items():
        traj_keys = set()
        probs = []
        for d in demos:
            max_reward = d.cumulative_reward
            max_trajectory_keys = []
            for k, r in d.possible_rewards.items():
                # A bit dangerous perhaps since comparing floats
                if math.isclose(r, max_reward):
                    max_trajectory_keys.append(k)

            w = 1 / len(max_trajectory_keys)

            traj_probs = {}
            for k in d.possible_rewards.keys():
                traj_keys.add(k)
                if k in max_trajectory_keys:
                    traj_probs[k] = w
                else:
                    traj_probs[k] = 0
            probs.append(traj_probs)

        traj_keys = list(traj_keys)
        aligned_probs = []
        for p in probs:
            ap = []
            for k in traj_keys:
                ap.append(p[k])
            aligned_probs.append(ap)

        aligned_probs = np.array(aligned_probs)
        M = len(aligned_probs)
        task_scores[i] = np.sum(aligned_probs * np.log2(M * aligned_probs / np.sum(aligned_probs, axis=0))) / M

    return task_scores


def inv_obs_traj_dist_vol_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    return {k : -1 * v for k,v in obs_trajectory_dist_volume_removal(experiments).items()}

def inv_reward_dist_volume_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    return {k : -1 * v for k,v in reward_dist_volume_removal(experiments).items()}

def inv_obs_feat_dist_volume_removal(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    return {k : -1 * v for k,v in obs_feat_dist_volume_removal(experiments).items()}

def inv_reward_dist_volume_removal_v2(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    return {k : -1 * v for k,v in reward_dist_volume_removal_v2(experiments).items()}

def inv_reward_dist_infomation_gain(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    return {k : -1 * v for k,v in information_gain(experiments).items()}

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

def chi_metric(experiments: Dict[int, List[TrajectoryResult]]) -> Dict[int, float]:
    task_scores = {}
    for i, trajectories in experiments.items():
        task_scores[i] = 0
    return task_scores

# TODO: Unique trajectories over all rollouts (like chi)

def task_trajectories(tasks: Dict[int, RIRLTask]) -> Dict[int, float]:
    task_scores = {}
    for i, task in tasks.items():
        task_scores[i] = task.num_trajectories()
    return task_scores

Metric = namedtuple("Metric", ["name", "func"])

METRICS = {
    "unique-trajectories": Metric("unique trajectories / sampled weights", unique_trajectories_metric),
    "unique-cumulative-features": Metric("unique cumulative features / sampled weights",
                                         unique_cumulative_features_metric),
    "dispersion": Metric("dispersion", dispersion_metric),
    "normed-dispersion": Metric("normed-dispersion", normed_dispersion_metric),
    "cos-dispersion": Metric("cos-dispersion", cos_dispersion_metric),
    "chi": Metric("Calinski-Harabasz Index", chi_metric),
    "num-task-trajectories": Metric("Number of Unique Possible Trajectories", task_trajectories),
    "obs-trajectory-distribution-volume-removal": Metric("Observed Trajectory Distribution Volume Removal", obs_trajectory_dist_volume_removal),
    "inv-obs-trajectory-distribution-volume-removal": Metric("Inverse Observed Trajectory Distribution Volume Removal", inv_obs_traj_dist_vol_removal),
    "reward-dist-volume-removal": Metric("Observed Trajectory Reward Distribution Volume Removal", reward_dist_volume_removal),
    "inv-reward-dist-volume-removal": Metric("Inverse Observed Trajectory Reward Distribution Volume Removal", inv_reward_dist_volume_removal),
    "obs-feat-dist-volume-removal": Metric("Observed Trajectory Feature Distribution Volume Removal", obs_feat_dist_volume_removal),
    "inv-obs-feat-dist-volume-removal": Metric("Inverse Observed Trajectory Feature Distribution Volume Removal", inv_obs_feat_dist_volume_removal),
    "reward-dist-volume-removal-v2": Metric("Observed Trajectory Reward Distribution Volume Removal (Ver. 2)", reward_dist_volume_removal_v2),
    "inv-reward-dist-volume-removal-v2": Metric("Inverse Observed Trajectory Reward Distribution Volume Removal (Ver. 2)", inv_reward_dist_volume_removal_v2),
    "reward-dist-information-gain-v2": Metric("Observed Trajectory Reward Distribution Information Gain", information_gain),
    "inv-reward-dist-information-gain-v2": Metric("Inverse Observed Trajectory Reward Distribution Information Gain", inv_reward_dist_infomation_gain),
}


def score_agent_distingushability(experiments: Dict[int, List[TrajectoryResult]], metric_key: str) -> Dict[int, float]:
    try:
        assert (metric_key in list(METRICS.keys()))
    except:
        raise RuntimeError(f"Invalid metric {metric_key} (valid metrics: {list(METRICS.keys())})")

    return METRICS[metric_key].func(experiments)