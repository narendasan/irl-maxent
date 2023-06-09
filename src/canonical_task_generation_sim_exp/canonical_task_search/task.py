from copy import deepcopy
import numpy as np
import math
from dataclasses import dataclass

class RIRLTask:
    def __init__(self, features, preconditions):
        self.num_actions, self.num_features = np.shape(features)
        self.actions = self._generate_action_space()
        self.action_features = np.array(features)
        self.action_preconditions = np.array(preconditions)
        self.states = self._generate_state_space()
        assert(np.equal(self.states[len(self.states) - 1], np.ones((self.num_actions))).all())
        self.state_features = self._generate_state_features_from_action_features()
        assert(self.state_features.shape == (self.states.shape[0], self.num_features))
        self.terminal_states = [self.states[len(self.states) - 1]]
        self.state_key_to_state_idx = {RIRLTask.state_to_key(s) : i for i, s in enumerate(self.states)}

    @staticmethod
    def state_to_key(x: np.array):
        assert(x.dtype == np.uint8)
        return hash(x.data.tobytes())

    def _generate_state_space(self):
        max_val = (2 ** self.num_actions)
        state_list = [np.array(list(map(int, np.binary_repr(i, width=self.num_actions)))) for i in range(max_val)]
        states = np.vstack(state_list)
        assert states.shape == (max_val, self.num_actions)

        return states.astype(np.uint8)

    def _generate_action_space(self):
        return np.array(range(self.num_actions))

    def _generate_state_features_from_action_features(self):
        return np.array([np.sum(self.action_features[np.where(taken_actions_at_state)], axis=0) for taken_actions_at_state in self.states])


    '''
    def num_trajectories(self):
        hashed_states = {RIRLTask.state_to_key(s) : s for s in self.states}
        start = self.states[0]
        end = self.terminal_states[0]

        def find_trajectories(s, e, visited, path):
            visited[RIRLTask.state_to_key(s)] = True
            path.append(s)

            if RIRLTask.state_to_key(s) == RIRLTask.state_to_key(e):
                return [deepcopy(path)]
            else:
                trajectories = []
                raw_state = hashed_states[RIRLTask.state_to_key(s)]
                pot_adj = [i for i, a in enumerate(raw_state) if a == 0]
                adj_a = [a for a in pot_adj if self.transition(s, a)[1] is not None]
                adj = []
                for a in adj_a:
                    new_s = deepcopy(s)
                    new_s[a] = 1
                    adj.append(new_s)

                for i in adj:
                    if visited[RIRLTask.state_to_key(i)] == False:
                        trajectories += find_trajectories(i, e, visited, path)

                path.pop()
                visited[RIRLTask.state_to_key(s)]= False
                return trajectories

        path = []
        visited = {s: False for s in hashed_states.keys()}
        trajectories = find_trajectories(start, end, visited, path)
        return len(trajectories)
    '''
    def num_trajectories(self):
        hashed_states = {RIRLTask.state_to_key(s) : s for s in self.states}
        visited = {s: False for s in hashed_states.keys()}
        start = self.states[0]
        end = self.terminal_states[0]

        path = []
        trajectories = []
        def find_trajectories(s, e, visited, path):
            visited[RIRLTask.state_to_key(s)] = True
            path.append(s)

            if RIRLTask.state_to_key(s) == RIRLTask.state_to_key(e):
                trajectories.append(deepcopy(path))
            else:
                raw_state = hashed_states[RIRLTask.state_to_key(s)]
                pot_adj = [i for i, a in enumerate(raw_state) if a == 0]
                adj_a = [a for a in pot_adj if self.transition(s, a)[1] is not None]
                adj = []
                for a in adj_a:
                    new_s = deepcopy(s)
                    new_s[a] = 1
                    adj.append(new_s)

                for i in adj:
                    if visited[RIRLTask.state_to_key(i)] == False:
                        find_trajectories(i, e, visited, path)
            path.pop()
            visited[RIRLTask.state_to_key(s)]= False

        find_trajectories(start, end, visited, path)
        return len(trajectories)

    def r_max(self):
        # THIS IS R_MAX AS LONG AS FEATURE WEIGHTS ARE NO GREATER THAN 1
        return np.sum(self.features)

    # @staticmethod
    def transition(self, s_from, a):

        satisfy_preconditions = [s_from[ap_idx] for ap_idx, ap in enumerate(self.action_preconditions[a]) if ap]

        # Action has been performed already
        if s_from[a] == 0 and all(satisfy_preconditions):
            s_to = deepcopy(s_from)
            s_to[a] = 1
            return 1.0, s_to.astype(np.uint8)
        else:
            return 0.0, None


    def __str__(self) -> str:
        return f"""
Task:
    Action Space: {self.num_actions}
    Features: \n{self.action_features}
"""

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    task = RIRLTask(features=np.random.random((5, 3)))
    print(task.actions, task.states)