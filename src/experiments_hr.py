# import python libraries
import os
import pdb
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle

# import functions
import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration
from maxent_irl import *
from assembly_tasks import *
from import_qualtrics import get_qualtrics_survey

# ----------------------------------------------- Load data ---------------------------------------------------- #

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
data_path = os.path.dirname(__file__) + "/data/"
# get_qualtrics_survey(dir_save_survey=data_path, survey_id=learning_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Learning (space).csv"
df = pd.read_csv(demo_path)


# pre-process feature value
def process_val(x):
    if x == "1 (No effort at all)":
        x = 1.1
    elif x == "7 (A lot of effort)":
        x = 6.9
    else:
        x = float(x)

    return x


# load user ratings
def load_features(data, user_idx, feature_idx, action_idx):
    fea_mat = []
    for j in action_idx:
        fea_vec = []
        for k in feature_idx:
            fea_col = k + str(j)
            fea_val = process_val(data[fea_col][user_idx])
            fea_vec.append(fea_val)
        fea_mat.append(fea_vec)
    return fea_mat


# ----------------------------------------------- Optimization -------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# --------------------------------------------- User information ------------------------------------------------ #

run_random_baseline = False

rank_features = False
scale_weights = False

predict_scores, random_scores = [], []
for user_id in [11, 12, 13]:

    user_id = str(user_id)
    print("=======================")
    print("Calculating preference for user:", user_id)

    idx = df.index[df['Q1'] == user_id][0]
    canonical_survey_actions = [0, 3, 1, 4, 2, 5]
    preferred_order = [df[q][idx] for q in ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6']]
    canonical_demo = [a for _, a in sorted(zip(preferred_order, canonical_survey_actions))]

    # user ratings for features
    canonical_q, complex_q = ["Q6_", "Q7_"], ["Q13_", "Q14_"]
    canonical_features = load_features(df, idx, canonical_q, [2, 4, 6, 3, 5, 7])
    complex_features = load_features(df, idx, complex_q, [3, 8, 15, 16, 4, 9, 10, 11])

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # initialize canonical task
    C = CanonicalTask(canonical_features)
    C.set_end_state(canonical_demo)
    C.enumerate_states()
    C.set_terminal_idx()
    if rank_features:
        C.convert_to_rankings()

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #
    sample_complex_demo = [1, 3, 5, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7]

    complex_survey_actions = [0, 4, 1, 5, 6, 7, 2, 3]
    action_counts = [1, 1, 4, 1, 4, 1, 4, 1]
    preferred_order = [df[q][idx] for q in ['Q15_1', 'Q15_2', 'Q15_3', 'Q15_4', 'Q15_5', 'Q15_6', 'Q15_7', 'Q15_8']]
    space_usage = [float(df[q][idx]) for q in ['Q22_1', 'Q22_2', 'Q22_3', 'Q22_4', 'Q22_5', 'Q22_6', 'Q22_7', 'Q22_8']]

    complex_demo = []
    for _, a in sorted(zip(preferred_order, complex_survey_actions)):
        complex_demo += [a]*action_counts[a]

    print(space_usage)
    print(complex_features)
    print(preferred_order)
    # initialize complex task
    X = ComplexTask(complex_features)
    X.set_end_state(sample_complex_demo)
    X.enumerate_states()
    X.set_terminal_idx()
    if rank_features:
        X.convert_to_rankings()

    complex_trajectories = get_trajectories(X.states, [complex_demo], X.transition)

    # using abstract features
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    # score for predicting the action based on transferred rewards based on abstract features
    load_path = data_path + "learned_models/q_values_" + user_id + ".p"
    qf_transfer = pickle.load(open(load_path, "rb"))
    # complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_abstract_features,
    #                                                                 complex_trajectories, optim, init)
    # # transfer rewards to complex task
    # transfer_rewards_abstract = complex_rewards_abstract
    #
    # # compute q-values for each state based on learned weights
    # qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract, X.terminal_idx)

    # predict
    predict_sequence, predict_score = predict_trajectory(qf_transfer, X.states, [complex_demo], X.transition,
                                                         sensitivity=0.0, consider_options=False)
    predict_scores.append(predict_score)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    if run_random_baseline:
        random_score = []
        for _ in range(100):
            # score for selecting actions based on random weights
            random_weights = np.random.rand(6)  # np.random.shuffle(canonical_weights_abstract)
            random_rewards_abstract = complex_abstract_features.dot(random_weights)
            qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards_abstract, X.terminal_idx)
            predict_sequence, r_score = predict_trajectory(qf_random, X.states, [complex_demo], X.transition,
                                                           sensitivity=0.0, consider_options=False)

            # # score for randomly selecting an action
            # predict_sequence, r_score = random_trajectory(X.states, [complex_demo], X.transition)

            random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)

np.savetxt(os.path.dirname(__file__) + "/results/study_hr/predict.csv", predict_scores)
if run_random_baseline:
    np.savetxt(os.path.dirname(__file__) + "/results/study_hr/random_weights.csv", random_scores)
