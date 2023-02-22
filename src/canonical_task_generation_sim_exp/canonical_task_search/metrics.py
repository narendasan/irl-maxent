from typing import List, Dict
import numpy as np
from collections import namedtuple

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
    "normed-dispersion": Metric("normed-dispersion", normed_dispersion_metric),
    "cos-dispersion": Metric("cos-dispersion", cos_dispersion_metric),
}


def score_agent_distingushability(experiments: Dict[int, List[TrajectoryResult]], metric_key: str) -> Dict[int, float]:
    try:
        assert (metric_key in list(METRICS.keys()))
    except:
        raise RuntimeError(f"Invalid metric {metric_key} (valid metrics: {list(METRICS.keys())})")

    return METRICS[metric_key].func(experiments)