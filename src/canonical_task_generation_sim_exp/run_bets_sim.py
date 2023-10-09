import pandas as pd
import numpy as np
from dask.distributed import LocalCluster, Client
import uuid
from copy import deepcopy
import pickle
import os
from rich.progress import track
import seaborn as sns
import matplotlib.pyplot as plt

from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask, CustomTask
from canonical_task_generation_sim_exp.lib.vi import value_iteration_numba as value_iteration
from canonical_task_generation_sim_exp.lib.irl import rollout_trajectory
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.simulate_user_demos import simulate_user
from canonical_task_generation_sim_exp.irl_maxent import optimizer as O
from canonical_task_generation_sim_exp.learn_reward_function import learn_reward_func
from canonical_task_generation_sim_exp.evaluate_results import evaluate_rf_acc

sns.set(font_scale=2, rc={'figure.figsize':(15,9)})

TARGET_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent/src/bets/new_target_tasks_df.pkl")
BEST_SOURCE_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent/src/bets/new_best_source_tasks_df.pkl")
WORST_SOURCE_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent/src/bets/new_worst_source_tasks_df.pkl")
TEST_USERS = pd.read_pickle("/Users/naren/Developer/py/irl-maxent/src/bets/new_test_users.pkl")

SOURCE_TASK_SIZES = [2,3,4,5,6,7,8]
TARGET_TASK_SIZES = [8,10,12,14,16]
FEAT_SIZES = [2,3,4,5]

def collect_user_demos(dask_client, user_df, source_task_df, target_tasks, kind):
    source_task_demos = []
    for id, s_task in track(source_task_df.iterrows(), description=f"Generating tasks for demo sim [{kind}]", total=len(source_task_df)):
        users = user_df.loc[user_df["feat_dim"] == s_task["feat_dim"]]

        s_feats = s_task["features"].tolist()
        s_preconditions = s_task["preconditions"].tolist()

        C = CustomTask(s_feats, s_preconditions)
        C.set_end_state(list(range(s_task["num_canonical_actions"])))
        C.enumerate_states()
        C.set_terminal_idx()
        C.set_transition_matrix()
        C.state_features = np.array([C.get_features(state) for state in C.states])

        for _, u in users.iterrows():
            source_task_demos.append((u["user_id"], np.array(u["weights"]), id, deepcopy(C), s_task["target_uid"], deepcopy(target_tasks[s_task["target_uid"]]), s_task["feat_dim"], C.num_actions, target_tasks[s_task["target_uid"]].num_actions))

    futures = dask_client.map(lambda s: simulate_user(s[1], s[3], s[5]), source_task_demos)
    sim_results = dask_client.gather(futures)

    demo_list = []
    for s, r in zip(source_task_demos, sim_results):
        print(f"========= {kind} ===========")
        print("Task:", s[4])
        print("User:", s[0])
        print("Canonical demo:", r[0])
        print("  Complex demo:", r[1])
        demo_list.append((s[0],s[2],s[4],s[6],s[7],s[8],r[0],r[1]))

    demo_df = pd.DataFrame(demo_list, columns=["user_id", "source_uid", "target_uid", "feat_dim", "num_canonical_actions", "num_complex_actions", "canonical_demo", "complex_demo"])
    demo_df.to_pickle(f"{kind}_bets_task_user_demos.pkl")

    return demo_df

def train(dask_client, source_task_df, user_demos, kind):
    training_list = []
    for _, demo in track(user_demos.iterrows(), description=f"Generating training tasks [{kind}]", total=len(user_demos)):
        s_task = source_task_df.loc[demo["source_uid"]]
        s_feats = s_task["features"].tolist()
        s_preconditions = s_task["preconditions"].tolist()

        C = CustomTask(s_feats, s_preconditions)
        C.set_end_state(list(range(demo["num_canonical_actions"])))
        C.enumerate_states()
        C.set_terminal_idx()
        C.set_transition_matrix()
        C.state_features = np.array([C.get_features(state) for state in C.states])
        canonical_demo = demo["canonical_demo"]

         # select initial distribution of weights
        #init = O.Constant(0.5)
        #weight_samples = np.random.uniform(0., 1., (50, demo["feat_dim"]))

        init = O.Uniform()  # Constant(0.5)

        training_list.append((demo["source_uid"], demo["user_id"], deepcopy(C), deepcopy(canonical_demo), deepcopy(init), None, demo["feat_dim"], demo["num_canonical_actions"]))

    futures = dask_client.map(lambda t: learn_reward_func(t[2], t[3], t[4], t[5], test_canonical=True), training_list)
    training_results = dask_client.gather(futures)

    learned_weights = []
    for a, r in zip(training_list, training_results):
        print(f"========={kind}==========")
        print("Task:", a[0])
        print("User:", a[1])
        print("Weights have been learned for the canonical task! Hopefully.")
        print("Weights -", r[0])
        print("Canonical task:")
        print("     demonstration -", a[3])
        if r[2] is not None:
            print("     predicted demo -", r[2])

        if r[1] is not None:
            print("predict (abstract) -", r[1])

        learned_weights.append((a[0], a[1], a[6], a[7], r[0], r[1]))

    learned_weights_df = pd.DataFrame(learned_weights, columns=["source_uid", "user_id", "feat_dim", "num_canonical_actions", "learned_weights", "canonical_task_acc"])
    learned_weights_df.to_pickle(f"{kind}_bets_learned_weights.pkl")

    return learned_weights_df

def eval(dask_client, target_tasks, learned_weights, user_demos, kind):
    eval_list = []
    for _, demo in track(user_demos.iterrows(), description=f"Generating eval tasks [{kind}]", total=len(user_demos)):
        user_id = demo["user_id"]
        source_uid = demo["source_uid"]
        target_uid = demo["target_uid"]
        feat_dim = demo["feat_dim"]
        num_canonical_actions = demo["num_canonical_actions"]
        num_complex_actions = demo["num_complex_actions"]
        canonical_demo = demo["canonical_demo"]
        complex_demo = demo["complex_demo"]
        learned_user_weights = learned_weights.loc[learned_weights["user_id"] == user_id]
        learned_user_weights = learned_user_weights.loc[learned_weights["source_uid"] == source_uid]["learned_weights"].item()
        complex_task = deepcopy(target_tasks[target_uid])

        eval_list.append((user_id, source_uid, target_uid, feat_dim, num_canonical_actions, num_complex_actions, complex_task, learned_user_weights, complex_demo))

    futures = dask_client.map(lambda e: evaluate_rf_acc(e[-3], e[-2], e[-1]), eval_list)
    eval_results = dask_client.gather(futures)

    rf_acc = []
    for a, r in zip(eval_list, eval_results):
        print("=======================")
        print(f"Task: {a[2]} (trained on: {a[1]})")
        print("User:", a[0])
        print(f" Avg: {r[1]}")
        print("\n")
        print("Complex task:")
        print("   demonstration -", a[-1])
        print("     predictions -", r[0])
        rf_acc.append((a[0], a[1], a[2], a[3], a[4], a[5], r[1], a[-1], r[0]))

    rf_acc_df = pd.DataFrame(rf_acc, columns=["user_id", "source_uid", "target_uid", "feat_dim", "num_canonical_actions", "num_complex_actions", "complex_task_acc", "complex_task_demo", "complex_task_predicted"])
    rf_acc_df.to_pickle(f"{kind}_bets_rf_acc.pkl")

    return rf_acc_df


def main():

    cluster = LocalCluster(
        processes=True,
        n_workers=24,
        threads_per_worker=1
    )

    dask_client = Client(cluster)

    user_df_data = []
    for f in FEAT_SIZES:
        for user in TEST_USERS[f][:5]:
            uid = uuid.uuid4()
            user_df_data.append((uid, f, user))
    user_df = pd.DataFrame(user_df_data, columns=["user_id", "feat_dim", "weights"])


    complex_tasks = {}
    for _, t_task in TARGET_TASKS_DF.iterrows():
        t_feat = np.array(t_task["features"])
        t_preconditions = np.array(t_task["preconditions"])

        complex_tasks[t_task["target_uid"]] = CustomTask(t_feat, t_preconditions)
        complex_tasks[t_task["target_uid"]].set_end_state(list(range(t_task["num_complex_actions"])))
        complex_tasks[t_task["target_uid"]].enumerate_states()
        complex_tasks[t_task["target_uid"]].set_terminal_idx()
        complex_tasks[t_task["target_uid"]].set_transition_matrix()
        complex_tasks[t_task["target_uid"]].state_features = np.array([complex_tasks[t_task["target_uid"]].get_features(state) for state in complex_tasks[t_task["target_uid"]].states])

    with open('bets_complex_tasks.pkl', 'wb') as handle:
        pickle.dump(complex_tasks, handle)

    if os.path.isfile("best_bets_task_user_demos.pkl"):
        best_user_demos_df = pd.read_pickle("best_bets_task_user_demos.pkl")
    else:
        best_user_demos_df = collect_user_demos(dask_client, user_df, BEST_SOURCE_TASKS_DF, complex_tasks, "best")

    if os.path.isfile("worst_bets_task_user_demos.pkl"):
        worst_user_demos_df = pd.read_pickle("worst_bets_task_user_demos.pkl")
    else:
        worst_user_demos_df = collect_user_demos(dask_client, user_df, WORST_SOURCE_TASKS_DF, complex_tasks, "worst")

    print(best_user_demos_df)
    print(worst_user_demos_df)

    if os.path.isfile("best_bets_learned_weights.pkl"):
        best_learned_weight_df = pd.read_pickle("best_bets_learned_weights.pkl")
    else:
        best_learned_weight_df = train(dask_client, BEST_SOURCE_TASKS_DF, best_user_demos_df, "best")

    if os.path.isfile("worst_bets_learned_weights.pkl"):
        worst_learned_weight_df = pd.read_pickle("worst_bets_learned_weights.pkl")
    else:
        worst_learned_weight_df = train(dask_client, WORST_SOURCE_TASKS_DF, worst_user_demos_df, "worst")

    print(best_learned_weight_df)
    print(worst_learned_weight_df)

    if os.path.isfile("best_bets_rf_acc.pkl"):
        best_rf_acc =  pd.read_pickle("best_bets_rf_acc.pkl")
    else:
        best_rf_acc = eval(dask_client, complex_tasks, best_learned_weight_df, best_user_demos_df, "best")

    if os.path.isfile("worst_bets_rf_acc.pkl"):
        worst_rf_acc =  pd.read_pickle("worst_bets_rf_acc.pkl")
    else:
        worst_rf_acc = eval(dask_client, complex_tasks, worst_learned_weight_df, worst_user_demos_df, "worst")

    print(best_rf_acc)
    print(worst_rf_acc)

    best_rf_acc_trimed = best_rf_acc.drop(['user_id', 'source_uid', "target_uid", "complex_task_demo", "complex_task_predicted"], axis=1)
    best_rf_acc_trimed = best_rf_acc_trimed.set_index(["feat_dim", "num_canonical_actions", "num_complex_actions"])
    worst_rf_acc_trimed = worst_rf_acc.drop(['user_id', 'source_uid', "target_uid", "complex_task_demo", "complex_task_predicted"], axis=1)
    worst_rf_acc_trimed = worst_rf_acc_trimed.set_index(["feat_dim", "num_canonical_actions", "num_complex_actions"])

    best_task_avg_acc_across_tasks = best_rf_acc_trimed.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    b_data = best_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_rf_acc_trimed.groupby(level=["feat_dim", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Feature Dimension", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Feature Dimension", "Number of Actions in Canonical Task", "complex_task_acc")

    #f, axes = plt.subplots(3, 1, figsize=(10, 24), sharex=True, sharey=True)
    #ax = axes.flat

    plot = sns.heatmap(
        data=b_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[0]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name})")
    plt.savefig("best_bets.png")

    plt.show()
    plt.clf()

    plot = sns.heatmap(
        data=w_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[2]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name})")

    plt.savefig("worst_bets.png")
    plt.show()
    plt.clf()

    best_task_avg_acc_across_tasks = best_rf_acc_trimed.groupby(level=["num_complex_actions", "num_canonical_actions"]).mean()
    best_task_avg_acc_across_tasks.index = best_task_avg_acc_across_tasks.index.set_names(["Number of Actions in Complex Task", "Number of Actions in Canonical Task",])
    b_data = best_task_avg_acc_across_tasks.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

    worst_task_avg_acc_across_tasks = worst_rf_acc_trimed.groupby(level=["num_complex_actions", "num_canonical_actions"]).mean()
    worst_task_avg_acc_across_tasks.index = worst_task_avg_acc_across_tasks.index.set_names(["Number of Actions in Complex Task", "Number of Actions in Canonical Task"])
    w_data = worst_task_avg_acc_across_tasks.reset_index().pivot("Number of Actions in Complex Task", "Number of Actions in Canonical Task", "complex_task_acc")

    #f, axes = plt.subplots(3, 1, figsize=(10, 24), sharex=True, sharey=True)
    #ax = axes.flat

    plot = sns.heatmap(
        data=b_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[0]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by best metric score ({METRICS[args.metric].name})")
    plt.savefig("best_bets_feat.png")

    plt.show()
    plt.clf()

    plot = sns.heatmap(
        data=w_data,
        annot=True,
        vmin=.8,
        vmax=1.,
        cmap=sns.color_palette("viridis", as_cmap=True),
        annot_kws={"size": 30}
        #ax=ax[2]
    )
    #plot.set(title=f"Avg. accuracy on complex tasks using canonical tasks\n selected by worst metric score ({METRICS[args.metric].name})")

    plt.savefig("worst_bets_feat.png")

    plt.show()
    plt.clf()



if __name__  == "__main__":
    main()