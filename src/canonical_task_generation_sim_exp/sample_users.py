import scipy.stats
import numpy as np
from typing import Tuple
import pandas as pd

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path

def sample_users_feat(feat_space_size: int, num_users: int = 10) -> np.array:
    shape = (num_users, feat_space_size)
    rng = scipy.stats.qmc.Halton(d=shape[1], scramble=False)
    users = rng.random(n=shape[0] + 1)[1:]  # Skip the first one which is always 0,0,0 when scramble is off
    return users

def create_user_archive(feat_space_range: Tuple[int, int], num_users: int = 10) -> pd.DataFrame:
    users = {}
    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        feat_users = sample_users_feat(f, num_users)
        for ui, u in enumerate(feat_users):
            users[(f, ui)] = (u, 0)

    user_feat = list(users.keys())
    user_idx = pd.MultiIndex.from_tuples(user_feat, name=["feat_dim", ""])
    user_set = list(users.values())
    user_df = pd.DataFrame(user_set, index=user_idx, columns=["users", "ph"])
    user_df.pop("ph")

    return user_df

def save_users(user_df: pd.DataFrame, args) -> None:
    p = out_path(args, kind="data", owner="user_archive")

    with (p / "user_archive.csv").open("w") as f:
        user_df.to_csv(f)


