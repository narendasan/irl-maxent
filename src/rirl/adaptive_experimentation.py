import numpy as np
from typing import Tuple, Dict
from task import RIRLTask
from agent import Agent

def to_key(x: np.array):
    assert(x.dtype == np.uint8)
    return hash(x.data.tobytes())

def find_s_max_s_min() -> Tuple[np.array, np.array]:
    """
    Find states s_max, s_min s.t. R(s_min) = R_min and R(s_max) = R_max
    """

def find_alpha() -> Dict[bytearray, float]:
    """
    Find for all states, alpha s.t. R(s) = alpha * R_min + (1 - alpha) * R_max
    """

def selection_heuristic():
    pass

def run_adapative_experimentation(experiments: Tuple[RIRLTask, Agent]):
    pass

