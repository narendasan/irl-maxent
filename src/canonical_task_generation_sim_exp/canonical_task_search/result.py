from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TrajectoryResult:
    trajectory: List[Tuple]
    num_ties: int
    cumulative_seen_features: np.array
    cumulative_reward: float
    cumulative_features_by_weights: float
    possible_rewards: List[float]
    valid_trajectories: Optional[List[List[Tuple]]]
    possible_cumulative_feats: List[np.array]

@dataclass
class TaskFeatsConditions:
    features: np.array
    preconditions: np.array
    score: float
    kind: str
