from typing import Tuple
import scipy.stats
import numpy as np

def sample_halton(shape: Tuple) -> np.array:
    rng = scipy.stats.qmc.Halton(d=shape[1], scramble=False)
    return rng.random(n=shape[0] + 1)[1:]  # Skip the first one which is always 0,0,0 when scramble is off


def sample_spherical(shape: Tuple) -> np.array:
    # phi = np.linspace(0, np.pi, 2000)
    # theta = np.linspace(0, 2 * np.pi, 4000)
    # x = 0.5 * np.outer(np.sin(theta), np.cos(phi)) + 0.5
    # y = 0.5 * np.outer(np.sin(theta), np.sin(phi)) + 0.5
    # z = 0.5 * np.outer(np.cos(theta), np.ones_like(phi)) + 0.5
    #
    # x = x.flatten()
    # x = x[np.newaxis, :]
    # x = x.T
    #
    # y = y.flatten()
    # y = y[np.newaxis, :]
    # y = y.T
    #
    # z = z.flatten()
    # z = z[np.newaxis, :]
    # z = z.T
    #
    # space = np.hstack((x, y, z))
    # sample = space[np.random.choice(np.arange(space.shape[0]), shape[0], replace=False), :]

    users = []
    while len(users) < shape[0]:
        sample = np.random.normal(loc=0., scale=1., size=shape[1])
        sample = sample.round(decimals=2)
        d = np.linalg.norm(sample)
        user = sample / d
        if all(user >= 0.) and all(user <= 1.) and list(user) not in users:
            users.append(list(user))

    sample = np.array(users)

    return sample


WEIGHT_SPACE = {
    "normal": lambda agents, feats: np.random.normal(loc=0.0, scale=1.0, size=(agents, feats)),
    "halton": lambda agents, feats: sample_halton(shape=(agents, feats)),
    "spherical": lambda agents, feats: sample_spherical(shape=(agents, feats))
}