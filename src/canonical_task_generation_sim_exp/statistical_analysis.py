import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from dataclasses import dataclass
from scipy.stats import f_oneway
import itertools

@dataclass
class MetricResult:
    result_name: str
    result_path: str
    task_archive_path: str


FILES = [
    MetricResult(
        "dispersion",
        "/Users/naren/Downloads/data/results/learned_rf_acc/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_dispersion-weight_space_spherical/score_spanning_learned_rf_acc.csv",
        "/Users/naren/Downloads/data/data/canonical_task_archive/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_dispersion-weight_space_spherical/score_spanning_task_archive.csv"
    ),
    MetricResult(
        "num-task-trajectories",
        "/Users/naren/Downloads/data/results/learned_rf_acc/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_num-task-trajectories-weight_space_spherical/score_spanning_learned_rf_acc.csv",
        "/Users/naren/Downloads/data/data/canonical_task_archive/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_num-task-trajectories-weight_space_spherical/score_spanning_task_archive.csv"
    ),
    MetricResult(
        "unique-cumulative-features",
        "/Users/naren/Downloads/data/results/learned_rf_acc/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/score_spanning_learned_rf_acc.csv",
        "/Users/naren/Downloads/data/data/canonical_task_archive/num_exp256-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/score_spanning_task_archive.csv"
    ),
    MetricResult(
        "unique-trajectories",
        "/Users/naren/Downloads/data/results/learned_rf_acc/num_exp256-weight_samples256-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-trajectories-weight_space_spherical/score_spanning_learned_rf_acc.csv",
        "/Users/naren/Downloads/data/data/canonical_task_archive/num_exp256-weight_samples256-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-trajectories-weight_space_spherical/score_spanning_task_archive.csv"
    ),
]

def process_data(score_spanning_task_acc_df: pd.DataFrame, canonical_task_archive: pd.DataFrame) -> pd.DataFrame:

    score_df = canonical_task_archive.reset_index()
    score_df["feat_dim"] = score_df.feat_dim.astype(str)
    score_df["num_actions"] = score_df.num_actions.astype(str)
    score_df["Feature Size / Number of Canonical Actions"] = score_df.feat_dim.str.cat(score_df.num_actions, sep="/")
    score_df["num_canonical_actions"] = score_df["num_actions"]
    score_df = score_df.drop(columns=["num_actions", "features", "preconditions", "feat_dim"])

    acc_df = score_spanning_task_acc_df.reset_index()
    acc_df["feat_dim"] = acc_df.feat_dim.astype(str)
    acc_df["num_canonical_actions"] = acc_df.num_canonical_actions.astype(str)
    acc_df["Feature Size / Number of Canonical Actions"] = acc_df.feat_dim.str.cat(acc_df.num_canonical_actions, sep="/")
    acc_df.set_index(["feat_dim", "num_canonical_actions",  "canonical_task_id", "num_complex_actions", "complex_task_id", "uid", "Feature Size / Number of Canonical Actions"], inplace=True)
    acc_df = acc_df.groupby(level=["feat_dim",  "num_canonical_actions",  "canonical_task_id", "num_complex_actions",  "complex_task_id", "Feature Size / Number of Canonical Actions"]).mean()
    acc_df = acc_df.reset_index()
    acc_df = acc_df.drop(columns=["feat_dim",  "num_canonical_actions"])

    data = {}
    for index, row in score_df.iterrows():
        relevant_data = acc_df.loc[(acc_df["Feature Size / Number of Canonical Actions"] == row["Feature Size / Number of Canonical Actions"]) & (acc_df["canonical_task_id"] == row["id"])]

        for i, rdr in relevant_data.iterrows():
            data_keys = (row["Feature Size / Number of Canonical Actions"], rdr["num_complex_actions"], rdr["complex_task_id"], row["score"])
            data_vals = (rdr["complex_task_acc"],)
            data[data_keys] = data_vals

    levels = list(data.keys())
    idx = pd.MultiIndex.from_tuples(levels, names=["feat_dim/num_canonical_actions", "num_complex_actions", "complex_task_id", "score"])
    score_acc = [[t[0]] for t in data.values()]

    results = pd.DataFrame(score_acc, index=idx, columns=["complex_task_acc"])
    return results


def one_way_anova(f: MetricResult):
    scores = pd.read_csv(f.result_path)
    tasks = pd.read_csv(f.task_archive_path)

    results_df = process_data(scores, tasks)
    #results_df = results_df.groupby(level=["feat_dim/num_canonical_actions", "num_complex_actions", "complex_"score"]).mean()

    f_can_dims = results_df.index.unique(level="feat_dim/num_canonical_actions")
    comp_dims = results_df.index.unique(level="num_complex_actions")

    for fc_sizes, comp_dim in itertools.product(f_can_dims, comp_dims):
        trial_df = results_df.xs((fc_sizes, comp_dim), level=["feat_dim/num_canonical_actions", "num_complex_actions"])

        trial_acc = {k: v["complex_task_acc"].tolist() for k,v in trial_df.groupby(level=["score"])}
        if fc_sizes == "4/8":
            print(f"Stats Results (feat/canonical sizes {fc_sizes}, complex sizes {comp_dim}): {f_oneway(*trial_acc.values())}")

def one_way_anova_alt(f: MetricResult):
    scores = pd.read_csv(f.result_path)
    tasks = pd.read_csv(f.task_archive_path)

    results_df = process_data(scores, tasks)
    #results_df = results_df.groupby(level=["feat_dim/num_canonical_actions", "num_complex_actions", "complex_"score"]).mean()

    f_can_dims = results_df.index.unique(level="feat_dim/num_canonical_actions")
    comp_dims = results_df.index.unique(level="num_complex_actions")

    for fc_sizes, comp_dim in itertools.product(f_can_dims, comp_dims):
        trial_df = results_df.xs((fc_sizes, comp_dim), level=["feat_dim/num_canonical_actions", "num_complex_actions"])
        trial_df = trial_df.reset_index()

        model = ols("""complex_task_acc ~ C(score)""", data=trial_df).fit()

        table = sm.stats.anova_lm(model, typ=2)
        if fc_sizes == "4/8":
            print(f"Stats Results (feat/canonical sizes {fc_sizes}")
            print(table)

def two_way_anova(f: MetricResult):
    scores = pd.read_csv(f.result_path)
    tasks = pd.read_csv(f.task_archive_path)

    results_df = process_data(scores, tasks)
    f_can_dims = results_df.index.unique(level="feat_dim/num_canonical_actions")

    for fc_sizes in f_can_dims:
        trial_df = results_df.xs((fc_sizes,), level=["feat_dim/num_canonical_actions"])
        trial_df = trial_df.reset_index()

        #perform three-way ANOVA
        model = ols("""complex_task_acc ~ C(score) + C(num_complex_actions) +
                    C(score):C(num_complex_actions)""", data=trial_df).fit()

        table = sm.stats.anova_lm(model, typ=2)
        if fc_sizes == "4/8":
            print(trial_df)
            print(f"Stats Results (feat/canonical sizes {fc_sizes}")
            print(table)

def main():
    for f in FILES:
        print(f"Stats for {f.result_name}")
        one_way_anova(f)
        one_way_anova_alt(f)
        two_way_anova(f)


if __name__ == "__main__":
    main()