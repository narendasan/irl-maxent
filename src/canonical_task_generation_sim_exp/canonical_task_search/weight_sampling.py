from typing import Tuple
import scipy.stats
import numpy as np

def sample_halton(shape: Tuple) -> np.array:
    rng = scipy.stats.qmc.Halton(d=shape[1], scramble=False)
    return rng.random(n=shape[0] + 1)[1:]  # Skip the first one which is always 0,0,0 when scramble is off


def sample_spherical(shape: Tuple) -> np.array:
    phi = np.linspace(0, np.pi, 2000)
    theta = np.linspace(0, 2 * np.pi, 4000)
    x = 0.5 * np.outer(np.sin(theta), np.cos(phi)) + 0.5
    y = 0.5 * np.outer(np.sin(theta), np.sin(phi)) + 0.5
    z = 0.5 * np.outer(np.cos(theta), np.ones_like(phi)) + 0.5

    x = x.flatten()
    x = x[np.newaxis, :]
    x = x.T

    y = y.flatten()
    y = y[np.newaxis, :]
    y = y.T

    z = z.flatten()
    z = z[np.newaxis, :]
    z = z.T

    space = np.hstack((x, y, z))
    sample = space[np.random.choice(np.arange(space.shape[0]), shape[0], replace=False), :]

    return sample


WEIGHT_SPACE = {
    "normal": lambda agents, feats: np.random.normal(loc=0.0, scale=1.0, size=(agents, feats)),
    "halton": lambda agents, feats: sample_halton(shape=(agents, feats)),
    "spherical": lambda agents, feats: sample_spherical(shape=(agents, feats))
}

def generate_agent_feature_weights(num_agents: int, num_feats: int, space: str) -> np.array:
    try:
        assert (space in list(WEIGHT_SPACE.keys()))
    except:
        raise RuntimeError(f"Invalid weight space {space} (valid weight spaces: {list(WEIGHT_SPACE.keys())})")

    return WEIGHT_SPACE[space](num_agents, num_feats)