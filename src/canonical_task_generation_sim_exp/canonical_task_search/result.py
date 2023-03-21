from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TrajectoryResult:
    trajectory: List[Tuple]
    num_ties: int
    cumulative_seen_features: np.array
    valid_trajectories: Optional[List[List[Tuple]]]

@dataclass
class TaskFeatsConditions:
    features: np.array
    preconditions: np.array
    score: float
    kind: str
