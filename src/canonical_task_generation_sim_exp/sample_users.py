import scipy.stats
import numpy as np
from typing import Tuple, Any
import pandas as pd

from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.weight_sampling import WEIGHT_SPACE

def sample_users_feat(num_agents: int, num_feats: int, space: str) -> np.array:
    try:
        assert (space in list(WEIGHT_SPACE.keys()))
    except:
        raise RuntimeError(f"Invalid weight space {space} (valid weight spaces: {list(WEIGHT_SPACE.keys())})")

    return WEIGHT_SPACE[space](num_agents, num_feats)

def create_user_archive(feat_space_range: Tuple[int, int], num_users: int = 10, weight_space: str = "normal") -> pd.DataFrame:
    users = {}
    for f in range(feat_space_range[0], feat_space_range[1] + 1):
        feat_users = sample_users_feat(num_users, f,weight_space)
        for ui, u in enumerate(feat_users):
            users[(f, ui)] = (u, 0)

    user_feat = list(users.keys())
    user_idx = pd.MultiIndex.from_tuples(user_feat, name=["feat_dim", ""])
    user_set = list(users.values())
    user_df = pd.DataFrame(user_set, index=user_idx, columns=["users", "ph"])
    user_df.pop("ph")

    return user_df

def load_users(args) -> pd.DataFrame:
    p = out_path(args, kind="data", owner="user_archive", load=True)
    user_df = pd.read_csv(p / "user_archive.csv", index_col=[0,1], converters={"users": serialization.from_space_sep_list})
    user_df["users"] = user_df["users"].apply(np.array)
    return user_df

def save_users(user_df: pd.DataFrame, args) -> None:
    user_df["users"] = user_df["users"].apply(np.array)
    p = out_path(args, kind="data", owner="user_archive")

    with (p / "user_archive.csv").open("w") as f:
        user_df.to_csv(f)


