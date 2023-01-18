# import functionsimport os

from irl_maxent import optimizer as O
# stochastic gradient descent optimizer
from maxent_irl import *
from toy_assembly import *

# import python libraries
import pickle
import numpy as np
from os.path import exists


accuracies = {}

for task_class in ["best", "random", "worst"]:
    FILE_SUFFIX = "exp10_feat3_metric_cos-dispersion_space_normal"
    accuracies[task_class] = {}

    with open(task_class + '_' + FILE_SUFFIX + ".pkl", "rb") as f:
        tasks = pickle.load(f)

    for action_space_size in range(2, 10):
        accuracies[task_class][action_space_size] = []
        for j, task_features in enumerate(tasks[action_space_size]):
            # ------------------------------------------------ Feature values --------------------------------------------------- #

            # canonical_features = [[0.837, 0.244, 0.282],
            #                     [0.212, 0.578, 0.018],
            #                     [0.712, 0.911, 0.418],
            #                     [0.462, 0.195, 0.882],
            #                     [0.962, 0.528, 0.618],
            #                     [0.056, 0.861, 0.218]]

            canonical_features = task_features

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

            # -------------------------------------------------- Experiment ----------------------------------------------------- #

            # select algorithm
            run_maxent = True
            run_bayes = False
            run_random_actions = False
            run_random_weights = False
            online_learning = False

            # algorithm parameters
            noisy_users = False
            map_estimate = True

            # debugging flags
            test_canonical = False
            test_complex = False

            # select samples
            n_train_samples = 50
            n_test_samples = 1

            # select initial distribution of weights
            init = O.Constant(0.5)
            weight_samples = np.random.uniform(0., 1., (n_train_samples, n_features))
            d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
            weight_samples = weight_samples / d

            # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
            optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

            # -------------------------------------------------- Load data ------------------------------------------------------ #

            # paths
            root_path = "data/"
            canonical_path = root_path + f"user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_demos_{j}.csv"
            if noisy_users:
                complex_path = root_path + "user_demos/complex_demos_adversarial.csv"
            else:
                complex_path = root_path + f"user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_demos_{j}.csv"

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
            if exists(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_trajectories_{j}.csv"):
                all_canonical_trajectories = pickle.load(open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_canonical_trajectories_{j}.csv", "rb"))
            else:
                all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

            # initialize actual task
            X = ComplexTask(complex_features)
            X.set_end_state(complex_actions)
            X.enumerate_states()
            X.set_terminal_idx()
            if exists(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_trajectories_{j}.csv"):
                all_complex_trajectories = pickle.load(open(f"data/user_demos/{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_complex_trajectories_{j}.csv", "rb"))
            else:
                all_complex_trajectories = X.enumerate_trajectories([complex_actions])

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
                canonical_features = np.array([C.get_features(state) for state in C.states])
                canonical_features /= np.linalg.norm(canonical_features, axis=0)
                complex_features = np.array([X.get_features(state) for state in X.states])
                complex_features /= np.linalg.norm(complex_features, axis=0)

                # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

                if run_maxent:
                    print("Training using Max-Entropy IRL ...")
                    _, canonical_weights = maxent_irl(C, canonical_features, canonical_trajectories, optim, init)

                elif run_bayes:
                    print("Training using Bayesian IRL ...")
                    posteriors = []
                    weight_priors = np.ones(n_train_samples) / n_train_samples
                    for n_sample in range(n_train_samples):
                        sample = weight_samples[n_sample]
                        likelihood_all_trajectories, _ = boltzman_likelihood(canonical_features, all_canonical_trajectories, sample)
                        likelihood_user_demo, demo_reward = boltzman_likelihood(canonical_features, np.array(canonical_trajectories), sample)
                        likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_trajectories)
                        bayesian_update = (likelihood_user_demo[0] * weight_priors[n_sample])

                        posteriors.append(bayesian_update)
                    posteriors = list(posteriors / np.sum(posteriors))

                    # select the MAP (maximum a posteriori) weight estimate
                    max_posterior = max(posteriors)
                    canonical_weights = weight_samples[posteriors.index(max_posterior)]

                else:
                    canonical_weights = None

                print("Weights have been learned for the canonical task! Hopefully.")
                print("Weights -", canonical_weights)
                weights.append(canonical_weights)

                # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

                if test_canonical:
                    canonical_rewards = canonical_features.dot(canonical_weights)
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
                        transferred_weights = [canonical_weights]
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
                            if run_bayes:
                                print("Online Prediction using Bayesian IRL ...")
                                p_score, predict_sequence, _ = online_predict_trajectory(X, complex_user_demo,
                                                                                    all_complex_trajectories,
                                                                                    transferred_weight,
                                                                                    complex_features,
                                                                                    weight_samples, priors=[],
                                                                                    sensitivity=0.0,
                                                                                    consider_options=False,
                                                                                    run_bayes=True)
                            elif run_maxent:
                                print("Online Prediction using Max Entropy IRL ...")
                                optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))
                                ol_optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))
                                p_score, predict_sequence, _ = online_predict_trajectory(X, complex_user_demo,
                                                                                    all_complex_trajectories,
                                                                                    transferred_weight,
                                                                                    complex_features,
                                                                                    weight_samples,
                                                                                    sensitivity=0.0,
                                                                                    consider_options=False,
                                                                                    run_maxent=True,
                                                                                    optim=ol_optim,
                                                                                    init=init)
                        else:
                            p_score, predict_sequence, _ = predict_trajectory(qf_transfer, X.states,
                                                                            complex_user_demo,
                                                                            X.transition,
                                                                            sensitivity=0.0,
                                                                            consider_options=False)
                        predict_score.append(p_score)

                    predict_score = np.mean(predict_score, axis=0)
                    print(f" Avg: {predict_score}")
                    predict_scores.append(predict_score)
                    accuracies[task_class][action_space_size].append(predict_score)

                    print("\n")
                    print("Complex task:")
                    print("   demonstration -", complex_user_demo)
                    print("     predictions -", predict_sequence)

                # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

                if test_complex:
                    complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_features,
                                                                                    complex_trajectories,
                                                                                    optim, init, eps=1e-2)

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
                        r_score, predict_sequence, _ = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition,
                                                                        sensitivity=0.0, consider_options=False)
                        random_score.append(r_score)

                    random_score = np.mean(random_score, axis=0)
                    random_scores.append(random_score)

            # -------------------------------------------------- Save results --------------------------------------------------- #

            save_path = "results/sim/"

            if run_bayes:
                np.savetxt(save_path + "weights" + str(n_users) + f"_norm_feat_bayes_adv_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", weights)
                np.savetxt(save_path + "predict" + str(n_users) + f"_norm_feat_bayes_adv_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", predict_scores)

            if run_maxent:
                np.savetxt(save_path + "weights" + str(n_users) + f"_norm_feat_maxent_online100_all_int_adv_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", weights)
                np.savetxt(save_path + "predict" + str(n_users) + f"_norm_feat_maxent_online100_all_int_adv_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", predict_scores)

            if run_random_actions:
                np.savetxt(save_path + "random" + str(n_users) + f"_norm_feat_actions_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", random_scores)

            if run_random_weights:
                np.savetxt(save_path + "random" + str(n_users) + f"_norm_feat_weights_adv_{task_class}_actions_{action_space_size}_{FILE_SUFFIX}_weights_{j}.csv", random_scores)

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(accuracies)

mean_accuracies = {}
for task_class in ["best", "random", "worst"]:
    mean_accuracies[task_class] = {}

    for action_space_size in range(2, 10):
        a = np.vstack(accuracies[task_class][action_space_size])
        mean_accuracies[task_class][action_space_size] = np.mean(a)

pp.pprint(mean_accuracies)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

best_mean_accuracies = mean_accuracies["best"]
random_mean_accuracies = mean_accuracies["random"]
worst_mean_accuracies = mean_accuracies["worst"]

metric_df = pd.DataFrame({
    "Action Space Size": list(best_mean_accuracies.keys()),
    f"Prediction accuracy on Complex Task \nfor using Best Cannonical Task for Action Space Size N": list(best_mean_accuracies.values()),
    f"Prediction accuracy on Complex Task \nfor using Random Cannonical Task for Action Space Size N": list(random_mean_accuracies.values()),
    f"Prediction accuracy on Complex \nfor using Worst Cannonical Task for Action Space Size N": list(worst_mean_accuracies.values())
})
metric_df.name = f"Action Space Size vs. MaxEnt Prediction Accuracy"

plot = sns.lineplot(x="Action Space Size",
                y=f"Prediction accuracy on complex task",
                hue="Task Class",
                data=pd.melt(metric_df,
                            ["Action Space Size"],
                            var_name="Task Class",
                            value_name=f"Prediction accuracy on complex task"))
plot.set(title=f"Action space vs. Prediction accuracy on complex task")
plt.savefig(f"action_space_vs_complex_prediction_acc_feat_space_size_{FILE_SUFFIX}.png")
plt.show()

with open(f"complex_prediction_accuracy_{FILE_SUFFIX}.pkl") as f:
    pickle.dump(mean_accuracies, f)

print("Done.")
