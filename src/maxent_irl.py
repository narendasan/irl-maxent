import numpy as np
from vi import value_iteration
from copy import deepcopy


# ------------------------------------------------ IRL functions ---------------------------------------------------- #

def get_trajectories(states, demonstrations, transition_function):
    trajectories = []
    for demo in demonstrations:
        s = states[0]
        trajectory = []
        for action in demo:
            p, sp = transition_function(s, action)
            s_idx, sp_idx = states.index(s), states.index(sp)
            trajectory.append((s_idx, action, sp_idx))
            s = sp
        trajectories.append(trajectory)

    return trajectories


def feature_expectation_from_trajectories(s_features, trajectories):
    n_states, n_features = s_features.shape

    fe = np.zeros(n_features)
    for t in trajectories:  # for each trajectory
        for s_idx, a, sp_idx in t:  # for each state in trajectory

            fe += s_features[sp_idx]  # sum-up features

    return fe / len(trajectories)  # average over trajectories


def initial_probabilities_from_trajectories(states, trajectories):
    n_states = len(states)
    prob = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        prob[t[0][0]] += 1.0  # increment starting state

    return prob / len(trajectories)  # normalize


def compute_expected_svf(task, p_initial, reward, max_iters, eps=1e-5):

    states, actions, terminal = task.states, task.actions, task.terminal_idx
    n_states, n_actions = len(states), len(actions)

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for i in range(2*n_states):
        za = np.zeros((n_states, n_actions))  # za: action partition function
        for s_idx in range(n_states):
            for a in actions:
                prob, sp = task.transition(states[s_idx], a)
                if sp:
                    sp_idx = task.states.index(sp)
                    if zs[sp_idx] > 0.0:
                        za[s_idx, a] += np.exp(reward[s_idx]) * zs[sp_idx]

        zs = za.sum(axis=1)
        zs[terminal] = 1.0

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, max_iters))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, max_iters):  # longest trajectory: n_states
        for sp_idx in range(n_states):
            parents = task.prev_states(states[sp_idx])
            if parents:
                for s in parents:
                    s_idx = states.index(s)
                    a = states[sp_idx][-1]
                    d[sp_idx, t] += d[s_idx, t - 1] * p_action[s_idx, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def compute_expected_svf_using_rollouts(task, reward, max_iters):
    states, actions, terminal = task.states, task.actions, task.terminal_idx
    n_states, n_actions = len(states), len(actions)

    qf, vf, _ = value_iteration(states, actions, task.transition, reward, terminal)
    svf = np.zeros(n_states)
    for _ in range(n_states):
        s_idx = 0
        svf[s_idx] += 1
        while s_idx not in task.terminal_idx:
            max_action_val = -np.inf
            candidates = []
            for a in task.actions:
                p, sp = task.transition(states[s_idx], a)
                if sp:
                    if qf[s_idx][a] > max_action_val:
                        candidates = [a]
                        max_action_val = qf[s_idx][a]
                    elif qf[s_idx][a] == max_action_val:
                        candidates.append(a)

            if not candidates:
                print("Error: No candidate actions from state", s_idx)

            take_action = np.random.choice(candidates)
            p, sp = task.transition(states[s_idx], take_action)
            s_idx = states.index(sp)
            svf[s_idx] += 1

    e_svf = svf/n_states

    return e_svf


def maxent_irl(task, s_features, trajectories, optim, omega_init, eps=1e-3):

    # states, actions = task.states, task.actions

    # number of actions and features
    n_states, n_features = s_features.shape

    # length of each demonstration
    _, demo_length, _ = np.shape(trajectories)

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(s_features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(task.states, trajectories)

    # gradient descent optimization
    omega = omega_init  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = s_features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf_using_rollouts(task, reward, demo_length)
        grad = e_features - s_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute delta for convergence check
        delta = np.max(np.abs(omega_old - omega))
        # print(delta)

    # re-compute per-state reward and return
    return s_features.dot(omega), omega


# ----------------------------------------- Bayesian inference functions -------------------------------------------- #

def get_feature_count(state_features, trajectories):
    feature_counts = []
    for traj in trajectories:
        feature_count = deepcopy(state_features[traj[0][0]])
        for t in traj:
            feature_count += deepcopy(state_features[t[2]])
        feature_counts.append(feature_count)

    return feature_counts


def boltzman_likelihood(state_features, trajectories, weights, rationality=0.99):
    n_states, n_features = np.shape(state_features)
    likelihood, rewards = [], []
    for traj in trajectories:
        feature_count = deepcopy(state_features[traj[0][0]])
        for t in traj:
            feature_count += deepcopy(state_features[t[2]])
        total_reward = rationality * weights.dot(feature_count)
        rewards.append(total_reward)
        likelihood.append(np.exp(total_reward))

    return likelihood, rewards


def custom_likelihood(task, trajectories, qf):
    demos = np.array(trajectories)[:, :, 1]
    likelihood = []
    for demo in demos:
        p, _, _ = predict_trajectory(qf, task.states, [demo], task.transition, sensitivity=0, consider_options=False)
        likelihood.append(np.mean(p))

    return likelihood


# ------------------------------------------------ MDP functions ---------------------------------------------------- #

def random_trajectory(states, demos, transition_function):
    """
    random predicted trajectory
    """

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    generated_sequence, score = [], []
    for take_action in demo:
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                candidates.append(a)

        if not candidates:
            print(s)

        options = list(set(candidates))
        predict_action = np.random.choice(options)
        if take_action in options:
            acc = 1/len(options)
        else:
            acc = 0.0
        score.append(acc)

        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return score, generated_sequence


def rollout_trajectory(qf, states, transition_function, remaining_actions, start_state=0):

    s = start_state
    available_actions = deepcopy(remaining_actions)
    generated_sequence = []
    while len(available_actions) > 0:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                if qf[s][a] > max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif qf[s][a] == max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if not candidates:
            print(s)
        take_action = np.random.choice(candidates)
        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence


def predict_trajectory(qf, states, demos, transition_function, sensitivity=0, consider_options=False):

    # assume the same starting state and available actions for all users
    demo = demos[0]  # TODO: for demo in demos:
    s, available_actions = 0, list(demo.copy())

    scores, predictions, options = [], [], []
    for take_action in demo:

        max_action_val = -np.inf
        candidates, applicants = [], []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1+sensitivity)*max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1-sensitivity)*max_action_val <= qf[s][a] <= (1+sensitivity)*max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        candidates = list(set(candidates))
        applicants = list(set(applicants))

        predictions.append(candidates)
        options.append(applicants)

        if consider_options and (len(candidates) < len(applicants)):
            score = [take_action in candidates]
        else:
            score = []
            for predict_action in candidates:
                score.append(predict_action == take_action)
        scores.append(np.mean(score))

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return scores, predictions, options


# ------------------------------------------------- Contribution ---------------------------------------------------- #

def online_predict_trajectory(task, demos, task_trajectories, traj_likelihoods, weights, features, samples, priors,
                              optim, init, sensitivity=0, consider_options=False):

    # assume the same starting state and available actions for all users
    demo = demos[0]
    s, available_actions = 0, list(demo.copy())
    transition_function = task.transition
    states = task.states
    _, n_features = np.shape(features)

    priors = np.ones(len(samples)) / len(samples)

    up_weights = []
    track_dist = []
    scores, predictions, options = [], [], []
    for step, take_action in enumerate(demo):

        # compute policy for current estimate of weights
        rewards = features.dot(weights)
        qf, _, _ = value_iteration(task.states, task.actions, task.transition, rewards, task.terminal_idx)

        # anticipate user action in current state
        max_action_val = -np.inf
        candidates, applicants = [], []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        candidates = list(set(candidates))
        applicants = list(set(applicants))
        predictions.append(candidates)
        options.append(applicants)

        int_seq = rollout_trajectory(qf, states, transition_function, available_actions, s)

        # calculate accuracy of prediction
        if consider_options and (len(candidates) < len(applicants)):
            score = [take_action in candidates]
        else:
            dist = []
            score = []
            for predict_action in candidates:
                dist.append(int_seq.index(take_action))
                score.append(predict_action == take_action)
        scores.append(np.mean(score))
        track_dist.append(np.mean(dist))

        # update weights based on correct user action
        future_actions = deepcopy(available_actions)

        # if np.mean(score) < 1.0:
        #     print("Check...")

        # infer intended user action
        prev_weights = deepcopy(weights)
        p, sp = transition_function(states[s], take_action)
        future_actions.remove(take_action)
        ro = rollout_trajectory(qf, states, transition_function, future_actions, states.index(sp))
        future_actions.append(take_action)
        intended_user_demo = [demo[:step] + [take_action] + ro]
        intended_trajectories = get_trajectories(states, intended_user_demo, transition_function)

        # compute set from which user picks the intended action
        # all_complex_trajectories = [traj for traj in task_trajectories if all(traj[:step, 1] == demo[:step])]
        # all_intended_trajectories = [traj for traj in task_trajectories if all(traj[:step+1, 1] == demo[:step+1])]
        # likelihood_intention = boltzman_likelihood(features, all_intended_trajectories, prev_weights)
        # intention_idx = likelihood_intention.index(max(likelihood_intention))
        # intended_trajectories = [all_intended_trajectories[intention_idx]]

        # update weights
        # # bayesian approach
        # n_samples = 1000
        # new_samples, posterior = [], []
        # for n_sample in range(n_samples):
        #     weight_idx = np.random.choice(range(len(samples)), size=1, p=priors)[0]
        #     complex_weights = samples[weight_idx]
        #     # likelihood_all_traj, _ = boltzman_likelihood(features, task_trajectories, complex_weights)
        #     likelihood_all_traj = traj_likelihoods[weight_idx]
        #     # likelihood_user_demo = custom_likelihood(task, intended_trajectories, qf)
        #     likelihood_user_demo, r = boltzman_likelihood(features, intended_trajectories, complex_weights)
        #     likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_traj)
        #     bayesian_update = (likelihood_user_demo * priors[n_sample])
        #
        #     new_samples.append(complex_weights)
        #     posterior.append(np.prod(bayesian_update))
        #
        # posterior = list(posterior / np.sum(posterior))
        # max_posterior = max(posterior)
        # weights = samples[posterior.index(max_posterior)]

        # max entropy approach
        init_weights = init(n_features)   # prev_weights
        _, new_weights = maxent_irl(task, features, intended_trajectories, optim, init_weights, eps=1e-2)
        weights = deepcopy(new_weights)

        # samples = deepcopy(new_samples)
        # priors = deepcopy(posterior)

        up_weights.append(weights)
        print("Updated weights from", prev_weights, "to", weights)

        # priors = priors / np.sum(priors)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return scores, predictions, options, up_weights, track_dist
