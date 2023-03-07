import numpy as np


def value_iteration(states, actions, transition, rewards, terminal_states, delta=1e-3):
    """
    Perform value iteration to calculate converged values for each state
    Args:
        states: list of all states
        actions: list of all actions
        transition: function that takes in current state and action, and return the next state and probability
        terminal_states: index of terminal states
        rewards: list of rewards for each state
        delta: error threshold

    Returns: q-value for each state-action pair, value for each state, optimal actions in each state

    """
    vf = {s: 0 for s in range(len(states))}  # values
    op_actions = {s: 0 for s in range(len(states))}  # optimal actions

    qf = {s: {a: 0 for a in actions} for s in range(len(states))}

    for i in range(100):
        vf_temp = {s: 0 for s in range(len(states))}

        for j_state in vf:
            max_action = -1
            max_action_val = -np.inf

            # Check if terminal state
            if j_state in terminal_states:
                vf_temp[j_state] = rewards[j_state]
                # qf[j_state][k_action] = 1
                continue

            for k_action in actions:
                prob_ns, ns = transition(states[j_state], k_action)
                qf[j_state][k_action] = rewards[j_state]
                if ns:
                    int_ns = states.index(ns)
                    qf[j_state][k_action] += prob_ns * vf[int(int_ns)]

                # Select max value v = max_a q(s, a)
                if qf[j_state][k_action] > max_action_val:
                    max_action = k_action
                    max_action_val = qf[j_state][k_action]

            # Update the value of the state
            vf_temp[j_state] = max_action_val

            # Simultaneously store the best action for the state
            op_actions[j_state] = max_action

        # After iterating over all states check if values have converged
        np_v = []
        np_v_temp = []
        for s in vf:
            np_v.append(vf[s])
            np_v_temp.append(vf_temp[s])
        np_v = np.array(np_v)
        np_v_temp = np.array(np_v_temp)
        change = np.linalg.norm((np_v - np_v_temp))
        vf = vf_temp
        if change < delta:
            # print("VI converged after %d iterations" % (i))
            break

    if change >= delta:
        print("VI did not converge after %d iterations (delta=%.2f)" % (i, change))

    return qf, vf, op_actions


from numba import njit
@njit
def value_iteration_numba(actions, trans_prob, trans_state, rewards, terminal_states, delta=1e-3):
    """
    Perform value iteration to calculate converged values for each state
    Args:
        # states: list of all states
        actions: list of all actions
        trans_prob:
        trans_state:
        terminal_states: index of terminal states
        rewards: list of rewards for each state
        delta: error threshold
        # transition: function that takes in current state and action, and return the next state and probability
    Returns: q-value for each state-action pair, value for each state, optimal actions in each state
    """

    # vf = {s: 0 for s in range(len(states))}  # values
    # op_actions = {s: 0 for s in range(len(states))}  # optimal actions
    #
    # qf = {s: {a: 0 for a in actions} for s in range(len(states))}

    vf = [0. for s in range(len(rewards))]  # values
    op_actions = [0. for s in range(len(rewards))]  # optimal actions

    qf = [[0. for a in actions] for s in range(len(rewards))]

    for i in range(100):
        # vf_temp = {s: 0 for s in range(len(states))}
        vf_temp = [0. for s in range(len(rewards))]

        # for j_state in vf:
        for j_state in range(len(vf)):
            max_action = -1
            max_action_val = -np.inf

            # Check if terminal state
            if j_state in terminal_states:
                vf_temp[j_state] = rewards[j_state]
                # qf[j_state][k_action] = 1
                continue

            for k_action in actions:
                # prob_ns, ns = transition[j_state][k_action]
                prob_ns = trans_prob[j_state][k_action]
                int_ns = trans_state[j_state][k_action]
                qf[j_state][k_action] = rewards[j_state]
                # if ns:
                if int_ns >= 0:
                    # int_ns = states.index(ns)
                    qf[j_state][k_action] += prob_ns * vf[int(int_ns)]

                # Select max value v = max_a q(s, a)
                if qf[j_state][k_action] > max_action_val:
                    max_action = k_action
                    max_action_val = qf[j_state][k_action]

            # Update the value of the state
            vf_temp[j_state] = max_action_val

            # Simultaneously store the best action for the state
            op_actions[j_state] = max_action

        # After iterating over all states check if values have converged
        np_v = []
        np_v_temp = []
        # for s in vf:
        for s in range(len(vf)):
            np_v.append(vf[s])
            np_v_temp.append(vf_temp[s])
        np_v = np.array(np_v)
        np_v_temp = np.array(np_v_temp)
        change = np.linalg.norm(np_v - np_v_temp)
        vf = vf_temp
        if change < delta:
            # print("VI converged after %d iterations" % i)
            break

    if change >= delta:
        print("VI did not converge after", i, "iterations (delta=", change, ")")

    return qf, vf, op_actions