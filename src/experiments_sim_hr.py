# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from assembly_tasks import *
from visualize import *

# import python libraries
import pickle
import numpy as np
import os
import pandas as pd


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


# -------------------------------------------------- Load data ------------------------------------------------------ #

# paths
root_path = "data/"
canonical_path = root_path + "user_demos/canonical_demos.csv"
complex_path = root_path + "user_demos/complex_demos_adversarial.csv"

canonical_demos = np.loadtxt(canonical_path).astype(int)
complex_demos = np.loadtxt(complex_path).astype(int)

canonical_features = [[0.837, 0.244, 0.282],
                      [0.212, 0.578, 0.018],
                      [0.712, 0.911, 0.418],
                      [0.462, 0.195, 0.882],
                      [0.962, 0.528, 0.618],
                      [0.056, 0.861, 0.218]]

complex_features = [[0.950, 0.033, 0.180],
                    [0.044, 0.367, 0.900],
                    [0.544, 0.700, 0.380],
                    [0.294, 0.145, 0.580],
                    [0.794, 0.478, 0.780],
                    [0.169, 0.811, 0.041],
                    [0.669, 0.256, 0.980],
                    [0.419, 0.589, 0.241],
                    [0.919, 0.922, 0.441],
                    [0.106, 0.095, 0.641]]

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
data_path = os.path.dirname(__file__) + "/data/"
# get_qualtrics_survey(dir_save_survey=data_path, survey_id=learning_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Learning.csv"
df = pd.read_csv(demo_path)

# -------------------------------------------------- Optimizer ------------------------------------------------------ #

# initialize optimization parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

# select experiment
run_maxent = True
run_bayes = not run_maxent
run_random_baseline = False
visualize = False

# initialize list of scores
predict_scores, random_scores = [], []
weights, decision_pts = [], []

canonical_actions = list(range(len(canonical_features)))
complex_actions = list(range(len(complex_features)))

# # initialize canonical task
# C = CanonicalTask(canonical_features)
# C.set_end_state(canonical_actions)
# C.enumerate_states()
# C.set_terminal_idx()
# # all_canonical_trajectories = pickle.load(open("data/user_demos/canonical_trajectories.csv", "rb"))
# all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])
#
# # initialize actual task
# X = ComplexTask(complex_features)
# X.set_end_state(complex_actions)
# X.enumerate_states()
# X.set_terminal_idx()
# # all_complex_trajectories = pickle.load(open("data/user_demos/complex_trajectories.csv", "rb"))
# all_complex_trajectories = X.enumerate_trajectories([complex_actions])

w_min, w_max = np.inf, -np.inf

for user_id in [11, 12, 13]:

    user_id = str(user_id)
    print("=======================")
    print("Calculating preference for user:", user_id)

    # user ratings for features
    idx = df.index[df['Q1'] == user_id][0]
    canonical_q, complex_q = ["Q6_", "Q7_"], ["Q13_", "Q14_"]
    canonical_features = load_features(df, idx, canonical_q, [2, 4, 6, 3, 5, 7])
    complex_features = load_features(df, idx, complex_q, [3, 8, 15, 16, 4, 9, 10, 11])

    # initialize canonical task
    canonical_survey_actions = [0, 3, 1, 4, 2, 5]
    preferred_order = [df[q][idx] for q in ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6']]
    canonical_demo = [a for _, a in sorted(zip(preferred_order, canonical_survey_actions))]

    C = CanonicalTask(canonical_features)
    C.set_end_state(canonical_actions)
    C.enumerate_states()
    C.set_terminal_idx()
    # all_canonical_trajectories = pickle.load(open("data/user_demos/canonical_trajectories.csv", "rb"))
    all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

    sample_complex_demo = [1, 3, 5, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7]
    complex_survey_actions = [0, 4, 1, 5, 6, 7, 2, 3]
    action_counts = [1, 1, 4, 1, 4, 1, 4, 1]
    preferred_order = [df[q][idx] for q in ['Q15_1', 'Q15_2', 'Q15_3', 'Q15_4', 'Q15_5', 'Q15_6', 'Q15_7', 'Q15_8']]
    complex_demo = []
    for _, a in sorted(zip(preferred_order, complex_survey_actions)):
        complex_demo += [a] * action_counts[a]

    X = ComplexTask(complex_features)
    X.set_end_state(sample_complex_demo)
    X.enumerate_states()
    X.set_terminal_idx()
    # all_complex_trajectories = pickle.load(open("data/user_demos/complex_trajectories.csv", "rb"))
    all_complex_trajectories = X.enumerate_trajectories([complex_actions])

    # canonical demonstrations
    canonical_user_demo = [canonical_demo]  # [list(canonical_demos[user_id])]
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)
    if visualize:
        visualize_rel_actions(C, canonical_user_demo[0], user_id, "canonical")

    # complex demonstrations (ground truth)
    complex_user_demo = [complex_demo]  # [list(complex_demos[user_id])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # state features
    canonical_features = np.array([C.get_features(state) for state in C.states])
    canonical_features /= np.linalg.norm(canonical_features, axis=0)
    complex_features = np.array([X.get_features(state) for state in X.states])
    complex_features /= np.linalg.norm(complex_features, axis=0)
    _, n_features = np.shape(complex_features)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    if run_bayes:
        print("Training ...")

        # initialize prior over weights
        n_samples = 1000
        samples, posteriors = [], []
        weight_priors = np.ones(n_samples)/n_samples
        max_likelihood = - np.inf
        max_reward = 0
        for n_sample in range(n_samples):
            u = np.random.uniform(0., 1., n_features)
            d = 1.  # np.sum(u)  # np.sum(u ** 2) ** 0.5
            canonical_weights_prior = u / d

            likelihood_all_trajectories, _ = boltzman_likelihood(canonical_features, all_canonical_trajectories,
                                                                 canonical_weights_prior)
            likelihood_user_demo, demo_reward = boltzman_likelihood(canonical_features, canonical_trajectories,
                                                                    canonical_weights_prior)
            likelihood_user_demo = likelihood_user_demo/np.sum(likelihood_all_trajectories)
            bayesian_update = (likelihood_user_demo[0] * weight_priors[n_sample])

            samples.append(canonical_weights_prior)
            posteriors.append(bayesian_update)

        posteriors = list(posteriors / np.sum(posteriors))

        max_posterior = max(posteriors)
        canonical_weights_abstract = samples[posteriors.index(max_posterior)]

    elif run_maxent:
        _, canonical_weights_abstract = maxent_irl(C, canonical_features,
                                                   canonical_trajectories,
                                                   optim, init)
        if min(canonical_weights_abstract) < w_min:
            w_min = min(canonical_weights_abstract)

        if max(canonical_weights_abstract) > w_max:
            w_max = max(canonical_weights_abstract)
    else:
        canonical_weights_abstract = None

    print("Weights have been learned for the canonical task! Hopefully.")
    print("Weights -", canonical_weights_abstract)
    weights.append(canonical_weights_abstract)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    # qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards_abstract, C.terminal_idx)
    # predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)
    #
    # print("\n")
    # print("Canonical task:")
    # print("     demonstration -", canonical_user_demo)
    # print("predict (abstract) -", predict_sequence_canonical)

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    if run_bayes or run_maxent:
        predict_score = []

        # ws = []
        # for _ in range(25):
        #     weight_idx = np.random.choice(range(len(samples)), size=1, p=posteriors)[0]
        #     complex_weights_abstract = samples[weight_idx]
        #     ws.append(complex_weights_abstract)

        # transfer rewards to complex task
        transfer_rewards_abstract = complex_features.dot(canonical_weights_abstract)

        # score for predicting the action based on transferred rewards based on abstract features
        qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                            X.terminal_idx)
        # predict_sequence, p_score, decisions = predict_trajectory(qf_transfer, X.states, complex_user_demo,
        #                                                           X.transition,
        #                                                           sensitivity=0.0,
        #                                                           consider_options=False)

        samples, priors = [], []
        predict_sequence, p_score, decisions = online_predict_trajectory(X, complex_user_demo,
                                                                         all_complex_trajectories,
                                                                         canonical_weights_abstract,
                                                                         complex_features,
                                                                         samples, priors,
                                                                         sensitivity=0.0,
                                                                         consider_options=False)
        predict_score.append(p_score)

        predict_score = np.mean(predict_score, axis=0)
        predict_scores.append(predict_score)
        # decision_pts.append(decisions)

        if visualize:
            visualize_rel_actions(X, complex_user_demo[0], user_id, "actual", predict_sequence, complex_user_demo[0])

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    # using true features
    # complex_state_features = np.array(X.states) / np.linalg.norm(X.states, axis=0)
    # complex_rewards_true, complex_weights_true = maxent_irl(X, complex_state_features, complex_trajectories,
    #                                                         optim, init, eps=1e-2)

    # using abstract features
    # complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_abstract_features,
    #                                                                 complex_trajectories,
    #                                                                 optim, init, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    if run_random_baseline:
        print("Assuming random weights ...")
        random_score = []

        # random_priors = 1 - priors
        # random_priors /= np.sum(random_priors)
        # for _ in range(25):
        #     weight_idx = np.random.choice(range(len(samples)), size=1, p=random_priors)[0]
        #     random_weights = samples[weight_idx]

        n_samples = 100
        max_likelihood = - np.inf
        for _ in range(n_samples):
            u = np.random.uniform(0., 1., n_features)
            d = 1.0  # np.sum(u)  # np.sum(u ** 2) ** 0.5
            random_weights = u / d  # np.random.shuffle(canonical_weights_abstract)

            random_rewards_abstract = complex_features.dot(random_weights)
            qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards_abstract, X.terminal_idx)
            predict_sequence, r_score, _ = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition,
                                                              sensitivity=0.0, consider_options=False)

            # predict_sequence, r_score, _ = predict_trajectory(X, optim, init,
            #                                                   qf_random, complex_user_demo,
            #                                                   sensitivity=0.0,
            #                                                   consider_options=False)

            # # score for randomly selecting an action
            # predict_sequence, r_score = random_trajectory(X.states, complex_user_demo, X.transition)

            random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)

    print("\n")
    print("Complex task:")
    print("   demonstration -", complex_user_demo)
    print("     predictions -", predict_sequence)

# -------------------------------------------------- Save results --------------------------------------------------- #

if run_bayes:
    # np.savetxt("results/decide19.csv", decision_pts)
    # np.savetxt("results/toy/weights19_normalized_features_bayesian.csv", weights)
    np.savetxt("results/study_hr/predict17_norm_feat_bayes_norm4.csv", predict_scores)

if run_maxent:
    # np.savetxt("results/decide19.csv", decision_pts)
    # np.savetxt("results/toy/weights19_normalized_features_bayesian.csv", weights)
    np.savetxt("results/study_hr/predict3_norm_feat_maxent_online.csv", predict_scores)

if run_random_baseline:
    np.savetxt("results/toy/random19_normalized_features_bayesian_new3.csv", random_scores)

print("Done.")
