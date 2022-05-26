import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# plotting style
sns.set(style="darkgrid", context="talk")

dir_path = os.path.dirname(__file__)
canonical_p = np.loadtxt(dir_path + "/data/complex_ratings_physical.csv")
canonical_m = np.loadtxt(dir_path + "/data/complex_ratings_mental.csv")

n_users, n_actions = np.shape(canonical_p)

X, Y1, Y2 = [], [], []
for a in range(n_actions):
    y1 = [r[a] for r in canonical_p]
    y2 = [r[a] for r in canonical_m]
    Y1 += y1
    Y2 += y2
    X += [a]*len(y1)
df_dict = {"Actions": X, "Physical Effort": Y1, "Mental Effort": Y2}
df = pd.DataFrame(df_dict)

# plt.figure()
# sns.boxplot(x="Actions", y="Physical Effort", data=df)
# plt.gcf().subplots_adjust(bottom=0.175)
# plt.gcf().subplots_adjust(left=0.15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("Actions", fontsize=24)
# plt.ylabel("Physical Effort", fontsize=24)
# # plt.savefig("figures/canonical_physical_ratings.png", bbox_inches='tight')
#
# plt.figure()
# sns.boxplot(x="Actions", y="Mental Effort", data=df)
# plt.gcf().subplots_adjust(bottom=0.175)
# plt.gcf().subplots_adjust(left=0.15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("Actions", fontsize=24)
# plt.ylabel("Mental Effort", fontsize=24)
# plt.savefig("figures/canonical_mental_ratings.png", bbox_inches='tight')
# plt.show()

w1_scores = np.loadtxt(dir_path + "/results/study_hr/ws1.csv")
w2_scores = np.loadtxt(dir_path + "/results/study_hr/ws2.csv")
w3_scores = np.loadtxt(dir_path + "/results/study_hr/ws3.csv")
w11_scores = np.loadtxt(dir_path + "/results/study_hr/ws11.csv")
w12_scores = np.loadtxt(dir_path + "/results/study_hr/ws12.csv")
w13_scores = np.loadtxt(dir_path + "/results/study_hr/ws13.csv")
prev_w1 = [0.02547568, 0.11715375, 0.02412226, 0.25402015, 0.35301439, 0.14604633]
prev_w2 = [0.29352558, 0.22193841, 0.6810284,  0.11812399, 0.15700865, 0.19552145]
prev_w3 = [0.38209842, 0.67074255, 0.65865852, 0.54495099, 0.21960905, 0.49454286]
prev_w11 = [0.45794899, 0.05177797, 0.53698477, 0.00756696, 0.44639235, 0.21404805]
prev_w12 = [0.19376387, 0.05651862, 0.19328036, 0.19683522, 0.22843809, 0.34729683]
prev_w13 = [0.2139996, 0.05585254, 0.09204319, 0.26017999, 0.15250731, 0.61760283]
w1_diff, w2_diff, w3_diff = [], [], []
w11_diff, w12_diff, w13_diff = [], [], []
for i in range(len(w1_scores)):
    wd1 = np.linalg.norm(prev_w1 - w1_scores[i])
    wd2 = np.linalg.norm(prev_w2 - w2_scores[i])
    wd3 = np.linalg.norm(prev_w3 - w3_scores[i])
    wd11 = np.linalg.norm(prev_w11 - w11_scores[i])
    wd12 = np.linalg.norm(prev_w12 - w12_scores[i])
    wd13 = np.linalg.norm(prev_w13 - w13_scores[i])
    prev_w1, prev_w2, prev_w3 = w1_scores[i], w2_scores[i], w3_scores[i]
    prev_w11, prev_w12, prev_w13 = w11_scores[i], w12_scores[i], w13_scores[i]
    w1_diff.append(wd1)
    w2_diff.append(wd2)
    w3_diff.append(wd3)
    w11_diff.append(wd11)
    w12_diff.append(wd12)
    w13_diff.append(wd13)
plt.plot(w1_diff, 'r', linewidth=2.5, alpha=0.95)
plt.plot(w2_diff, 'g', linewidth=2.5, alpha=0.95)
plt.plot(w3_diff, 'b', linewidth=2.5, alpha=0.95)
plt.plot(w11_diff, 'r--', linewidth=2.5, alpha=0.95)
plt.plot(w12_diff, 'g--', linewidth=2.5, alpha=0.95)
plt.plot(w13_diff, 'b--', linewidth=2.5, alpha=0.95)
plt.show()

file_path = dir_path + "/results/study_hr/"
predict1_scores = np.loadtxt(file_path + "predict6_norm_feat_maxent_online_maxent.csv")
predict2_scores = np.loadtxt(file_path + "predict6_norm_feat_maxent.csv")
random1_scores = np.loadtxt(file_path + "random6_norm_feat_weights.csv")
random2_scores = np.loadtxt(file_path + "random6_norm_feat_actions.csv")

n_users, n_steps = np.shape(predict1_scores)

# check statistical difference
predict1_users = list(np.sum(predict1_scores, axis=1)/n_steps)
predict2_users = list(np.sum(predict2_scores, axis=1)/n_steps)
random1_users = list(np.sum(random1_scores, axis=1)/n_steps)
random2_users = list(np.sum(random2_scores, axis=1)/n_steps)
print("Random actions:", stats.ttest_rel(predict1_users, random1_users))
print("Online actions:", np.mean(predict1_users), np.mean(predict2_users),
      stats.ttest_rel(predict1_users, predict2_users))

# plt.figure()
# X1 = predict_users + random1_users
# Y = ["canonical weights"]*n_users + ["random weights"]*n_users
# df_dict = {"Y": X1, "X": Y}
# df = pd.DataFrame(df_dict)
# sns.barplot(x="X", y="Y", data=df)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# plt.title("Over all time steps")
# plt.savefig("figures/results19_random_weights.png", bbox_inches='tight')

# predict_users_new, random1_users_new, utility = [], [], []
# for i in range(n_users):
#     predict_new = [score for j, score in enumerate(predict_scores[i]) if decision_pts[i][j]]
#     random1_new = [score for j, score in enumerate(random1_scores[i]) if decision_pts[i][j]]
#     n_steps_new = len(predict_new)
#     utility.append(n_steps_new/n_steps)
#     predict_users_new.append(np.sum(predict_new)/n_steps_new)
#     random1_users_new.append(np.sum(random1_new)/n_steps_new)
# print("Random weights:", stats.ttest_rel(predict_users_new, random1_users_new))

# plt.figure()
# X2 = predict_users_new + random1_users_new
# df_dict = {"Y": X2, "X": Y}
# df = pd.DataFrame(df_dict)
# sns.barplot(x="X", y="Y", data=df)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# plt.title("Over all time steps")
# plt.savefig("figures/results19_random_weights_new.png", bbox_inches='tight')
# plt.show()

# accuracy at each time steps
predict1_accuracy = np.sum(predict1_scores, axis=0)/n_users
predict2_accuracy = np.sum(predict2_scores, axis=0)/n_users
random1_accuracy = np.sum(random1_scores, axis=0)/n_users
random2_accuracy = np.sum(random2_scores, axis=0)/n_users
steps = list(range(len(predict1_accuracy)))

plt.figure(figsize=(9, 5))

X, Y1, Y2 = [], [], []
# for i in range(n_users):
#     X += steps
#     Y1 += list(predict_scores[i, :])
#     Y2 += list(random2_scores[i, :])
# df1 = pd.DataFrame({"Time step": X, "Accuracy": Y1})
# df2 = pd.DataFrame({"Time step": X, "Accuracy": Y2})
# sns.lineplot(data=df2, x="Time step", y="Accuracy", color="r", linestyle="--", alpha=0.9)
# sns.lineplot(data=df1, x="Time step", y="Accuracy", color="g", linewidth=4, alpha=0.9)
plt.plot(steps, random2_accuracy, 'r:', linewidth=4.5, alpha=0.95)
plt.plot(steps, random1_accuracy, 'y-.', linewidth=4.5, alpha=0.95)
plt.plot(steps, predict2_accuracy, 'b--', linewidth=4.5, alpha=0.95)
plt.plot(steps, predict1_accuracy, 'g', linewidth=4.5, alpha=0.95)
plt.xlim(-1, 10)
plt.ylim(-0.1, 1.1)
plt.xticks(steps, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time step", fontsize=22)
plt.ylabel("Accuracy", fontsize=22)
# plt.title("Action prediction using personalized priors", fontsize=22)
plt.gcf().subplots_adjust(bottom=0.175)
plt.legend(["random actions", "random weights", "max-entropy", "max-entropy (online)"],
           fontsize=20, ncol=2, loc=8)
# plt.legend(["personalized prior", "online (corrected steps)", "online (all time steps)"],
#            fontsize=20, ncol=1, loc=4)
plt.show()
# plt.savefig("figures/sim/results17_norm_feat_online_adv.png", bbox_inches='tight')

# plt.figure()
# Y = list(predict_scores[:, 0]) + uniform_users
# X = ["proposed"]*n_users + ["uniform weights"]*n_users
# sns.barplot(X, Y, palette=['g', 'y'], ci=68)
# plt.ylim(-0.1, 1.1)
# plt.ylabel("Accuracy")
# plt.gcf().subplots_adjust(left=0.15)
# plt.savefig("figures/results11_timestep1.jpg", bbox_inches='tight')
# plt.show()

# Sensitivity

# predict1_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity2.csv")
# predict2_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity5.csv")
# predict3_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity10.csv")
#
# # accuracy at each time steps
# predict1_accuracy = np.sum(predict1_scores, axis=0)/n_users
# predict2_accuracy = np.sum(predict2_scores, axis=0)/n_users
# predict3_accuracy = np.sum(predict3_scores, axis=0)/n_users
# steps = range(1, len(predict_accuracy)+1)
#
# plt.figure(figsize=(10, 5))
# plt.plot(steps, predict_accuracy, 'g', linewidth=3.5)
# plt.plot(steps, predict1_accuracy, 'b-.', linewidth=3.5)
# plt.plot(steps, predict2_accuracy, 'r--', linewidth=3.5)
# plt.plot(steps, predict3_accuracy, 'y:', linewidth=3.5)
# plt.ylim(-0.1, 1.1)
# plt.xticks(steps)
# plt.xlabel("Time step")
# plt.ylabel("Accuracy")
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.legend(["proposed", "2%", "5%", "10%"], loc=4)
# plt.show()
# # plt.savefig("figures/results11_sensitivity.jpg", bbox_inches='tight')
