import pandas as pd
import numpy as np
from dask.distributed import LocalCluster, Client
import uuid
from copy import deepcopy
import pickle
from rich.progress import track


from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask


from canonical_task_generation_sim_exp.simulated_tasks.assembly_task import CanonicalTask, ComplexTask
from canonical_task_generation_sim_exp.lib.vi import value_iteration_numba as value_iteration
from canonical_task_generation_sim_exp.lib.irl import rollout_trajectory
from canonical_task_generation_sim_exp.lib import serialization
from canonical_task_generation_sim_exp.lib.arguments import parser, out_path
from canonical_task_generation_sim_exp.simulate_user_demos import simulate_user
from canonical_task_generation_sim_exp.irl_maxent import optimizer as O
from canonical_task_generation_sim_exp.learn_reward_function import learn_reward_func


TARGET_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent-2_2_exp/src/bets/target_tasks_df.pkl")
BEST_SOURCE_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent-2_2_exp/src/bets/best_source_tasks_df.pkl")
WORST_SOURCE_TASKS_DF = pd.read_pickle("/Users/naren/Developer/py/irl-maxent-2_2_exp/src/bets/worst_source_tasks_df.pkl")
TEST_USERS = pd.read_pickle("/Users/naren/Developer/py/irl-maxent-2_2_exp/src/bets/test_users.pkl")

SOURCE_TASK_SIZES = [2,3,4,5,6,7,8]
TARGET_TASK_SIZES = [8,10,12,14,16]
FEAT_SIZES = [2,3,4,5]

def collect_user_demos(dask_client, user_df, source_task_df, target_tasks, kind):
    source_task_demos = []
    for id, s_task in track(source_task_df.iterrows(), description=f"Generating tasks for demo sim [{kind}]", total=len(source_task_df)):
        users = user_df.loc[user_df["feat_dim"] == s_task["feat_dim"]]

        s_feats = np.array(s_task["features"])
        s_preconditions = np.array(s_task["preconditions"])

        canonical_task = CanonicalTask(s_feats, s_preconditions)

        for _, u in users.iterrows():
            source_task_demos.append((u["user_id"], np.array(u["weights"]), id, deepcopy(canonical_task), s_task["target_uid"], deepcopy(target_tasks[s_task["target_uid"]]), s_task["feat_dim"], canonical_task.num_actions, target_tasks[s_task["target_uid"]].num_actions))

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
        s_feats = np.array(s_task["features"])
        s_preconditions = np.array(s_task["preconditions"])

        canonical_task = CanonicalTask(s_feats, s_preconditions)
        canonical_demo = demo["canonical_demo"]

         # select initial distribution of weights
        init = O.Constant(0.5)
        weight_samples = np.random.uniform(0., 1., (50, demo["feat_dim"]))
        d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
        weight_samples = weight_samples / d
        training_list.append((demo["source_uid"], demo["user_id"], deepcopy(canonical_task), deepcopy(canonical_demo), deepcopy(init), deepcopy(weight_samples), demo["feat_dim"], demo["num_canonical_actions"]))
        learn_reward_func(canonical_task=canonical_task, canonical_demo=canonical_demo, init=init, weight_samples=weight_samples, test_canonical=True)

    #training_results = [learn_reward_func(canonical_task=t[2], canonical_demo=t[3], init=t[4], weight_samples=t[5], test_canonical=True) for t in training_list[:1]]
    #futures = dask_client.map(lambda t: learn_reward_func(t[2], t[3], t[4], t[5], test_canonical=True), training_list)
    #training_results = dask_client.gather(futures)

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

# def eval(dask_client, target_tasks, learned_weights, user_demos, kind):
#     eval_list = []
#     for _, demo in user_demos:


#     futures = dask_client.map(lambda e: evaluate_rf_acc(e[2], e[3], e[4]), eval_list)
#     eval_results = dask_client.gather(futures)

#     rf_acc = {}
#     for a, r in zip(eval_list, eval_results):
#         print("=======================")
#         print("Task:", a[0])
#         print("User:", a[1])
#         print(f" Avg: {r[1]}")
#         print("\n")
#         print("Complex task:")
#         print("   demonstration -", a[-1])
#         print("     predictions -", r[0])
#         rf_acc[(feat_size, canonical_action_space_size, complex_action_space_size, a[0], a[1])] = r

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

        complex_tasks[t_task["target_uid"]] = ComplexTask(t_feat, t_preconditions)

    with open('bets_complex_tasks.pkl', 'wb') as handle:
        pickle.dump(complex_tasks, handle)

    best_user_demos_df = collect_user_demos(dask_client, user_df, BEST_SOURCE_TASKS_DF, complex_tasks, "best")
    worst_user_demos_df = collect_user_demos(dask_client, user_df, WORST_SOURCE_TASKS_DF, complex_tasks, "worst")

    print(best_user_demos_df)

    best_learned_weight_df = train(dask_client, BEST_SOURCE_TASKS_DF, best_user_demos_df, "best")
    worst_learned_weight_df = train(dask_client, WORST_SOURCE_TASKS_DF, worst_user_demos_df, "worst")




if __name__  == "__main__":
    main()