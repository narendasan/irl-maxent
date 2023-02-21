from typing import Tuple
import scipy.stats
import numpy as np

from canonical_task_generation_sim_exp.lib.weight_sampling import WEIGHT_SPACE

def generate_agent_feature_weights(num_agents: int, num_feats: int, space: str) -> np.array:
    try:
        assert (space in list(WEIGHT_SPACE.keys()))
    except:
        raise RuntimeError(f"Invalid weight space {space} (valid weight spaces: {list(WEIGHT_SPACE.keys())})")

    return WEIGHT_SPACE[space](num_agents, num_feats)