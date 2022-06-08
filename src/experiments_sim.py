# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from toy_assembly import *

# import python libraries
import pickle
import numpy as np
from os.path import exists


# ------------------------------------------------ Feature values --------------------------------------------------- #

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

_, n_features = np.shape(complex_features)

adverse_features = [[0.950, 0.033, 0.180, 0.777],
                    [0.044, 0.367, 0.900, 0.152],
                    [0.544, 0.700, 0.380, 0.652],
                    [0.294, 0.145, 0.580, 0.402],
                    [0.794, 0.478, 0.780, 0.902],
                    [0.169, 0.811, 0.041, 0.090],
                    [0.669, 0.256, 0.980, 0.590],
                    [0.419, 0.589, 0.241, 0.340],
                    [0.919, 0.922, 0.441, 0.840],
                    [0.106, 0.095, 0.641, 0.215]]

weights_adverse = [[0.60, 0.20, 0.20, 0.60],
                   [0.80, 0.10, 0.10, 0.01],
                   [0.20, 0.60, 0.20, 0.60],
                   [0.10, 0.80, 0.10, 0.01],
                   [0.20, 0.20, 0.60, 0.60],
                   [0.10, 0.10, 0.80, 0.01],
                   [0.40, 0.40, 0.20, 0.20],
                   [0.40, 0.20, 0.40, 0.20],
                   [0.20, 0.40, 0.40, 0.20],
                   [0.40, 0.30, 0.30, 0.40],
                   [0.30, 0.40, 0.30, 0.40],
                   [0.50, 0.30, 0.20, 0.00],
                   [0.50, 0.20, 0.30, 0.00],
                   [0.30, 0.50, 0.20, 0.80],
                   [0.20, 0.50, 0.30, 0.80],
                   [0.30, 0.20, 0.50, 0.15],
                   [0.20, 0.30, 0.50, 0.80]]

# -------------------------------------------------- Experiment ----------------------------------------------------- #

# select algorithm
run_maxent = True
run_bayes = False
run_random_actions = False
run_random_weights = False
online_learning = True

# algorithm parameters
noisy_users = False
adverse_users = True
map_estimate = True
custom_prob = False

# debugging flags
test_canonical = False
test_complex = False

# select samples
n_train_samples = 1000
n_test_samples = 100

# select initial distribution of weights
init = O.Constant(0.5)
if exists("data/user_demos/weight_samples.csv"):
    weight_samples = np.loadtxt("data/user_demos/weight_samples.csv")
else:
    weight_samples = np.random.uniform(0., 1., (n_train_samples, n_features))
    d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
    weight_samples = weight_samples / d


# choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# -------------------------------------------------- Load data ------------------------------------------------------ #

# paths
root_path = "data/"
canonical_path = root_path + "user_demos/canonical_demos.csv"
if noisy_users:
    complex_path = root_path + "user_demos/complex_demos_adversarial.csv"
elif adverse_users:
    complex_path = root_path + "user_demos/adverse_demos.csv"
else:
    complex_path = root_path + "user_demos/complex_demos.csv"

# user demonstrations
canonical_demos = np.loadtxt(canonical_path).astype(int)
complex_demos = np.loadtxt(complex_path).astype(int)

n_users, _ = np.shape(canonical_demos)

# ------------------------------------------------------------------------------------------------------------------- #

# initialize list of scores
predict_scores, random_scores = [], []
weights, decision_pts = [], []

# assembly task actions
canonical_actions = list(range(len(canonical_features)))
complex_actions = list(range(len(complex_features)))

# initialize canonical task
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_actions)
C.enumerate_states()
C.set_terminal_idx()
all_canonical_trajectories = []
# if exists("data/user_demos/canonical_trajectories.csv"):
#     all_canonical_trajectories = pickle.load(open("data/user_demos/canonical_trajectories.csv", "rb"))
# else:
#     all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])
canonical_features = np.array([C.get_features(state) for state in C.states])
canonical_features /= np.linalg.norm(canonical_features, axis=0)

# initialize actual task
X = ComplexTask(complex_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()
all_complex_trajectories = []
# if exists("data/user_demos/complex_trajectories.csv"):
#     all_complex_trajectories = pickle.load(open("data/user_demos/complex_trajectories.csv", "rb"))
# else:
#     all_complex_trajectories = X.enumerate_trajectories([complex_actions])
complex_features = np.array([X.get_features(state) for state in X.states])
complex_features /= np.linalg.norm(complex_features, axis=0)

X2 = ComplexTask(adverse_features)
X2.set_end_state(complex_actions)
X2.enumerate_states()
X2.set_terminal_idx()
complex_features_adverse = np.array([X2.get_features(state) for state in X2.states])
complex_features_adverse /= np.linalg.norm(complex_features_adverse, axis=0)

if not exists("data/user_demos/adverse_demos.csv"):
    seqs = []
    for wa in weights_adverse:
        reward_adverse = complex_features_adverse.dot(wa)
        qf2, _, _ = value_iteration(X2.states, X2.actions, X2.transition, reward_adverse, X2.terminal_idx)
        seq = rollout_trajectory(qf2, X2.states, X2.transition, list(complex_demos[0]), 0)
        seqs.append(seq)
    np.savetxt("data/user_demos/adverse_demos.csv", seqs)

complex_likelihoods = []
# if custom_prob:
#     if exists("data/user_demos/custom_likelihoods.csv") and custom_prob:
#         complex_likelihoods = np.loadtxt("data/user_demos/custom_likelihoods.csv")
#         complex_qf = np.loadtxt("data/user_demos/complex_q_values.csv")
#     else:
#         complex_qf = []
#         for complex_weights in weight_samples:
#             save_path = "data/user_demos/custom_likelihoods.csv"
#             r = complex_features.dot(complex_weights)
#             qf, _, _ = value_iteration(X.states, X.actions, X.transition, r, X.terminal_idx)
#             likelihood = custom_likelihood(X, all_complex_trajectories, qf)
#             complex_likelihoods.append(likelihood)
#             complex_qf.append(qf)
#         np.savetxt("data/user_demos/custom_likelihoods.csv", complex_likelihoods)
#         np.savetxt("data/user_demos/complex_q_values.csv", complex_qf)
# else:
#     if exists("data/user_demos/complex_likelihoods.csv"):
#         complex_likelihoods = np.loadtxt("data/user_demos/complex_likelihoods.csv")
#     else:
#         for complex_weights in weight_samples:
#             likelihood, _ = boltzman_likelihood(complex_features, all_complex_trajectories, complex_weights)
#             complex_likelihoods.append(likelihood)
#         np.savetxt("data/user_demos/complex_likelihoods.csv", complex_likelihoods)


# loop over all users
for i in range(len(canonical_demos)):

    print("=======================")
    print("User:", i)

    # canonical demonstrations
    canonical_user_demo = [list(canonical_demos[i])]
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)

    # complex demonstrations (ground truth)
    complex_user_demo = [list(complex_demos[i])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # state features
    # canonical_features = np.array([C.get_features(state) for state in C.states])
    # canonical_features /= np.linalg.norm(canonical_features, axis=0)
    # complex_features = np.array([X.get_features(state) for state in X.states])
    # complex_features /= np.linalg.norm(complex_features, axis=0)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    if run_maxent:
        print("Training using Max-Entropy IRL ...")
        # canonical_weights = []
        # for _ in range(n_test_samples):
        init_weights = init(n_features)
        _, canonical_weight = maxent_irl(C, canonical_features, canonical_trajectories, optim, init_weights)
        #     canonical_weights.append(canonical_weight)

    elif run_bayes:
        print("Training using Bayesian IRL ...")
        posteriors, entropies = [], []
        weight_priors = np.ones(len(weight_samples)) / len(weight_samples)
        for n_sample in range(len(weight_samples)):
            sample = weight_samples[n_sample]
            likelihood_all_trajectories, _ = boltzman_likelihood(canonical_features, all_canonical_trajectories, sample)
            likelihood_user_demo, _ = boltzman_likelihood(canonical_features, canonical_trajectories, sample)
            # r = canonical_features.dot(sample)
            # qf, _, _ = value_iteration(C.states, C.actions, C.transition, r, C.terminal_idx)
            # likelihood_all_trajectories = custom_likelihood(C, all_canonical_trajectories, qf)
            # likelihood_user_demo = custom_likelihood(C, canonical_trajectories, qf)
            likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_trajectories)
            bayesian_update = likelihood_user_demo * weight_priors[n_sample]

            # p = likelihood_all_trajectories / np.sum(likelihood_all_trajectories)
            # entropy = np.sum(p*np.log(p))

            posteriors.append(np.prod(bayesian_update))
            entropies.append(np.sum(np.log(likelihood_user_demo)))

        posteriors = list(posteriors / np.sum(posteriors))

        # select the MAP (maximum a posteriori) weight estimate
        max_posterior = max(posteriors)
        canonical_weight = weight_samples[posteriors.index(max_posterior)]
        # max_entropy = max(entropies)
        # canonical_weights = weight_samples[entropies.index(max_entropy)]
        # all_max_posteriors = [idx for idx, p in enumerate(posteriors) if p == max_posterior]
        # all_max_entropies = [e for idx, e in enumerate(entropies) if idx in all_max_posteriors]
        # max_entropy = max(all_max_entropies)
        # canonical_weights = weight_samples[all_max_posteriors[all_max_entropies.index(max_entropy)]]

    else:
        canonical_weight = None

    print("Weights have been learned for the canonical task! Hopefully.")
    print("Weights -", canonical_weight)
    weights.append(canonical_weight)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    if test_canonical:
        canonical_rewards = canonical_features.dot(canonical_weight)
        qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
        predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)

        print("\n")
        print("Canonical task:")
        print("     demonstration -", canonical_user_demo)
        print("predict (abstract) -", predict_sequence_canonical)

    # ---------------------------------------- Testing: Predict complex --------------------------------------------- #

    if run_bayes or run_maxent:
        print("Testing ...")

        if map_estimate:
            transferred_weights = [canonical_weight]
        elif run_maxent:
            transferred_weights = canonical_weight
        elif run_bayes:
            weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples, p=posteriors)
            transferred_weights = weight_samples[weight_idx]
        else:
            transferred_weights = []

        predict_score = []
        for transferred_weight in transferred_weights:
            # transfer rewards to complex task
            transfer_rewards_abstract = complex_features.dot(transferred_weight)

            # compute policy for transferred rewards
            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                                X.terminal_idx)

            # score for predicting user action at each time step
            if online_learning:
                init_weights = init(n_features)

                p_score, predict_sequence, _, _, _ = online_predict_trajectory(X, complex_user_demo,
                                                                               all_complex_trajectories,
                                                                               complex_likelihoods,
                                                                               transferred_weight,
                                                                               complex_features,
                                                                               complex_features_adverse,
                                                                               weight_samples, [],
                                                                               optim, init,
                                                                               sensitivity=0.0,
                                                                               consider_options=False)
            else:
                p_score, predict_sequence, _ = predict_trajectory(qf_transfer, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
            predict_score.append(p_score)

        predict_score = np.mean(predict_score, axis=0)
        predict_scores.append(predict_score)

        print("\n")
        print("Complex task:")
        print("   demonstration -", complex_user_demo)
        print("     predictions -", predict_sequence)

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    if test_complex:
        init_weights = init(n_features)
        complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_features,
                                                                        complex_trajectories,
                                                                        optim, init_weights, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    if run_random_actions:
        # score for randomly selecting an action
        r_score, predict_sequence = random_trajectory(X.states, complex_user_demo, X.transition)
        random_scores.append(r_score)

    elif run_random_weights:
        print("Testing for random weights ...")

        # random_priors = 1 - priors
        # random_priors /= np.sum(random_priors)
        # weight_idx = np.random.choice(range(len(samples)), size=n_test_samples, p=random_priors)[0]

        weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples)
        random_weights = weight_samples[weight_idx]

        random_score = []
        max_likelihood = - np.inf
        for n_sample in range(n_test_samples):
            random_weight = random_weights[n_sample]
            random_rewards = complex_features.dot(random_weight)
            qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards, X.terminal_idx)
            # r_score, predict_sequence, _ = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition,
            #                                                   sensitivity=0.0, consider_options=False)
            if online_learning:
                init_weights = random_weight
                r_score, predict_sequence, _ = online_predict_trajectory(X, complex_user_demo,
                                                                         all_complex_trajectories,
                                                                         complex_likelihoods,
                                                                         random_weight,
                                                                         complex_features,
                                                                         weight_samples, [],
                                                                         optim, init_weights,
                                                                         sensitivity=0.0,
                                                                         consider_options=False)
            else:
                r_score, predict_sequence, _ = predict_trajectory(qf_random, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
            random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)

# -------------------------------------------------- Save results --------------------------------------------------- #

save_path = "results/sim/"

if run_bayes:
    np.savetxt(save_path + "weights" + str(n_users) + "_norm_feat_bayes_ent.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_norm_feat_bayes_ent.csv", predict_scores)

if run_maxent:
    np.savetxt(save_path + "weights" + str(n_users) + "_norm_feat_maxent_online_maxent_add_corr.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_norm_feat_maxent_online_maxent_add_corr.csv", predict_scores)

if run_random_actions:
    np.savetxt(save_path + "random" + str(n_users) + "_norm_feat_actions_adv.csv", random_scores)

if run_random_weights:
    np.savetxt(save_path + "random" + str(n_users) + "_norm_feat_weights_online_maxent.csv", random_scores)

print("Done.")
