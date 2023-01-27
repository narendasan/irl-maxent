from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class TrajectoryResult:
    trajectory: List[Tuple]
    num_ties: int
    cumulative_seen_features: np.array

@dataclass
class TaskFeatsConditions:
    features: np.array
    preconditions: np.array
    score: float
