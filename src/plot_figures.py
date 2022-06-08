import os
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from import_qualtrics import get_qualtrics_survey

# ---------------------------------------------------- Result ------------------------------------------------------- #

plot_subjective = False
plot_time = False
plot_accuracy = True

# plotting style
sns.set(style="darkgrid", context="talk")

# -------------------------------------------------- Load data ------------------------------------------------------ #

dir_path = os.path.dirname(__file__)
file_path = dir_path + "/results/corl/"
predict1_scores = np.loadtxt(file_path + "predict10_maxent_online_uni_new_rand.csv")
predict2_scores = np.loadtxt(file_path + "predict10_maxent.csv")
random1_scores = np.loadtxt(file_path + "random10_weights_online_uni_new.csv")
random2_scores = np.loadtxt(file_path + "random10_actions.csv")

# ------------------------------------------------- Time taken ------------------------------------------------------ #

# compute result for user idle time
times = pd.read_csv(file_path + "wait_times.csv", header=None)
reactive_times = times[1]
proactive_times = times[2]
print("Reactive:", np.mean(reactive_times), stats.sem(reactive_times))
print("Proactive:", np.mean(proactive_times), stats.sem(proactive_times))
print("T-test:", stats.ttest_rel(reactive_times, proactive_times))

# plot result for user idle time
if plot_time:
    x = list(reactive_times) + list(proactive_times)
    y = ["reactive "]*len(reactive_times) + ["proactive"]*len(proactive_times)
    plt.figure(figsize=(4, 4.5))
    sns.boxplot(y, x, width=0.6)
    plt.ylim(150, 210)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Time taken (secs)", fontsize=22)
    plt.gcf().subplots_adjust(left=0.275)
    # plt.show()
    # plt.savefig("figures/corl/idle_time.png", bbox_inches='tight')

# --------------------------------------------- Subjective response ------------------------------------------------- #

# download data from qualtrics
execution_survey_id = "SV_29ILBswADgbr79Q"
data_path = os.path.dirname(__file__) + "/data/"
# get_qualtrics_survey(dir_save_survey=data_path, survey_id=execution_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Execution.csv"
df = pd.read_csv(demo_path)

# users to consider for evaluation
users = [6, 7, 8, 9, 10, 14, 19, 20, 21, 22, 23]
user_idx = [df.index[df["Q0"] == str(user)][0] for user in users]
reactive_first_users = [6, 7, 8, 9, 10, 14]
proactive_first_users = [19, 20, 21, 22, 23]
reactive_first_user_idx = [df.index[df["Q0"] == str(user)][0] for user in reactive_first_users]
proactive_first_user_idx = [df.index[df["Q0"] == str(user)][0] for user in proactive_first_users]
condition1_q = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]
condition2_q = ["Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30"]

if plot_subjective:
    for q_idx in range(len(condition1_q)):
        x1 = df[condition1_q[q_idx]].iloc[reactive_first_user_idx]
        x2 = df[condition2_q[q_idx]].iloc[proactive_first_user_idx]

        y1 = df[condition2_q[q_idx]].iloc[reactive_first_user_idx]
        y2 = df[condition1_q[q_idx]].iloc[proactive_first_user_idx]

        x = list(x1) + list(x2)
        y = list(y1) + list(y2)

        x = list(map(int, x))
        y = list(map(int, y))

        print(condition1_q[q_idx], np.mean(x), np.mean(y), stats.ttest_rel(x, y))

# --------------------------------------------- Action anticipation ------------------------------------------------- #

# predict1_scores = [s for i, s in enumerate(predict1_scores) if i in [0, 1, 5, 6, 7, 9]]
# predict2_scores = [s for i, s in enumerate(predict2_scores) if i in [0, 1, 5, 6, 7, 9]]
# random1_scores = [s for i, s in enumerate(random1_scores) if i in [0, 1, 5, 6, 7, 9]]
# random2_scores = [s for i, s in enumerate(random2_scores) if i in [0, 1, 5, 6, 7, 9]]

n_users, n_steps = np.shape(predict1_scores)

# check statistical difference
predict1_users = list(np.sum(predict1_scores, axis=1)/n_steps)
predict2_users = list(np.sum(predict2_scores, axis=1)/n_steps)
random1_users = list(np.sum(random1_scores, axis=1)/n_steps)
random2_users = list(np.sum(random2_scores, axis=1)/n_steps)
print("Random actions:", stats.ttest_rel(predict1_users, random1_users))
print("Online actions:", np.mean(predict1_users), stats.sem(predict1_users),
      np.mean(predict2_users), stats.sem(predict2_users),
      stats.ttest_rel(predict1_users, predict2_users))

# accuracy over all users at each time step
predict1_accuracy = np.sum(predict1_scores, axis=0)/n_users
predict2_accuracy = np.sum(predict2_scores, axis=0)/n_users
random1_accuracy = np.sum(random1_scores, axis=0)/n_users
random2_accuracy = np.sum(random2_scores, axis=0)/n_users
steps = np.array(range(len(predict1_accuracy))) + 1.0

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
plt.xlim(0, 18)
plt.ylim(-0.1, 1.1)
plt.xticks(steps, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time step", fontsize=22)
plt.ylabel("Accuracy", fontsize=22)
# plt.title("Action prediction using personalized priors", fontsize=22)
plt.gcf().subplots_adjust(bottom=0.175)
plt.legend(["random actions", "random weights", "initial estimate", "proposed (online)"], fontsize=20, ncol=2, loc=8)
plt.show()
# plt.savefig("figures/corl/online_accuracy.png", bbox_inches='tight')
