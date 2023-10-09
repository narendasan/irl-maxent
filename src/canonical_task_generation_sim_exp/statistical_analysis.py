import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from dataclasses import dataclass
from scipy.stats import f_oneway
import itertools
from typing import List
import numpy as np

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

best_worst_random_data = {

}

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

def one_way_anova_best_on_best(files: List[MetricResult]):
    processed_data = {m.result_name: process_data(pd.read_csv(m.result_path), pd.read_csv(m.task_archive_path)) for m in files}

    f_can_dims = processed_data["dispersion"].index.unique(level="feat_dim/num_canonical_actions")
    comp_dims = processed_data["dispersion"].index.unique(level="num_complex_actions")

    for fc_sizes, comp_dim in itertools.product(f_can_dims, comp_dims):
        best_scores = []
        names = []
        for (name, results_df) in processed_data.items():
            names.append(name)
            trial_df = results_df.xs((fc_sizes, comp_dim), level=["feat_dim/num_canonical_actions", "num_complex_actions"])
            trial_df = trial_df.reset_index()

            best_scores.append(trial_df[trial_df["score"] == trial_df["score"].max()]["complex_task_acc"].tolist())


        if fc_sizes == "4/8":
            print(f"Stats Results (feat/canonical sizes {fc_sizes}, complex sizes {comp_dim}): {f_oneway(*best_scores)}")
            print(sum([len(l) for l in best_scores]))
            for n, l in zip(names, best_scores):
                x = np.array(l)
                print(n, x.mean(), x.std())


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

# def three_way_anova():
#     tasks = pd.read_csv("/Users/naren/Downloads/data/data/canonical_task_archive/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/search_results_task_archive.csv")
#     best_task_scores = pd.read_csv("/Users/naren/Downloads/data/results/learned_rf_acc/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/best_learned_rf_acc.csv")
#     random_task_scores = pd.read_csv("/Users/naren/Downloads/data/results/learned_rf_acc/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/random_learned_rf_acc.csv")
#     worst_task_scores = pd.read_csv("/Users/naren/Downloads/data/results/learned_rf_acc/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/worst_learned_rf_acc.csv")

#     random_task_scores.groupby(columns=["feat_dim",  "num_canonical_actions"])

#     feat_sizes = best_task_scores["feat_dim"].unique()
#     can_as = best_task_scores["num_canonical_actions"].unique()
#     comp_as = best_task_scores["num_complex_actions"].unique()

#     print(tasks)
#     print(random_task_scores)

#     data = []

#     for (f, can, comp) in itertools.product(feat_sizes, can_as, comp_as):
#         best_df =  best_task_scores[(best_task_scores["feat_dim"] == f) & (best_task_scores["num_canonical_actions"] == can) & (best_task_scores["num_canonical_actions"] == comp)]
#         for _, r in best_df.iterrows():
#             score = tasks[(tasks["feat_dim"] == f) & (tasks["num_actions"] == can) & (tasks["kind"] == "best")]["score"].item()
#             data.append([f, can, comp, score, r["complex_task_id"], r["uid"], r["complex_task_acc"], "best"])

#         random_df =  random_task_scores[(random_task_scores["feat_dim"] == f) & (random_task_scores["num_canonical_actions"] == can) & (random_task_scores["num_canonical_actions"] == comp)]
#         for _, r in random_df.iterrows():
#             #????
#             score = tasks[(tasks["feat_dim"] == f) & (tasks["num_actions"] == can) & (tasks["kind"] == "random")]["score"].mean()
#             data.append([f, can, comp, score, r["complex_task_id"], r["uid"], r["complex_task_acc"], "random"])

#         worst_df =  worst_task_scores[(worst_task_scores["feat_dim"] == f) & (worst_task_scores["num_canonical_actions"] == can) & (worst_task_scores["num_canonical_actions"] == comp)]
#         for _, r in worst_df.iterrows():
#             score = tasks[(tasks["feat_dim"] == f) & (tasks["num_actions"] == can) & (tasks["kind"] == "worst")]["score"].item()
#             data.append([f, can, comp, score, r["complex_task_id"], r["uid"], r["complex_task_acc"], "worst"])

#     df = pd.DataFrame(data)
#     df.columns = ["feat_dim", "num_canonical_actions", "num_complex_actions", "score", "complex_task_id", "uid", "complex_task_acc", "kind"]

#     model = ols("""complex_task_acc ~ C(score) + C(num_complex_actions) + C(feat_dim) +
#                     C(score):C(num_complex_actions) +  C(feat_dim):C(num_complex_actions) +
#                     C(score):C(feat_dim) +  C(score):C(num_complex_actions):C(feat_dim) """, data=df).fit()

#     table = sm.stats.anova_lm(model, typ=3)
#     print(df)
#     print(table)

def three_way_anova_alt():
    tasks = pd.read_csv("/Users/naren/Downloads/data/data/canonical_task_archive/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/search_results_task_archive.csv")
    search_task_results = pd.read_csv("/Users/naren/Downloads/data/results/learned_rf_acc/num_exp512-weight_samples128-max_canonical_action_space_size8-max_complex_action_space_size16-max_feat_size5-max_exp_len100-metric_unique-cumulative-features-weight_space_spherical/search_results_learned_rf_acc.csv")

    print(tasks)
    print(search_task_results)

    data = []

    for _, r in search_task_results.iterrows():
        row = list(r) + [tasks[(tasks["id"] == r["canonical_task_id"]) & (tasks["feat_dim"] == r["feat_dim"]) & (tasks["num_actions"] == r["num_canonical_actions"])]["score"], tasks[(tasks["id"] == r["canonical_task_id"]) & (tasks["feat_dim"] == r["feat_dim"]) & (tasks["num_actions"] == r["num_canonical_actions"])]["kind"]]


    df = pd.DataFrame(data)
    df.columns = search_task_results + ["score", "kind"]

    model = ols("""complex_task_acc ~ C(score) + C(num_complex_actions) + C(feat_dim) +
                    C(score):C(num_complex_actions) +  C(feat_dim):C(num_complex_actions) +
                    C(score):C(feat_dim) +  C(score):C(num_complex_actions):C(feat_dim) """, data=df).fit()

    table = sm.stats.anova_lm(model, typ=3)
    print(df)
    print(table)


def main():
    #for f in FILES:
        #print(f"Stats for {f.result_name}")
        #one_way_anova(f)
        #one_way_anova_alt(f)
        #two_way_anova(f)

    one_way_anova_best_on_best(FILES)
    #three_way_anova()
    #three_way_anova_alt()
    #UNBALANCED SAMPLE VALID?

if __name__ == "__main__":
    main()