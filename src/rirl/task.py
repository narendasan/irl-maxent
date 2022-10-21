from copy import deepcopy
import numpy as np

class RIRLTask:
    def __init__(self, features):
        self.num_actions, self.num_features = np.shape(features)
        self.actions = self._generate_action_space()
        self.action_features = np.array(features)
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

    def r_max(self):
        # THIS IS R_MAX AS LONG AS FEATURE WEIGHTS ARE NO GREATER THAN 1
        return np.sum(self.features)

    @staticmethod
    def transition(s_from, a):
        # Action has been performed already
        if s_from[a] == 1:
            return 0.0, None

        else:
            s_to = deepcopy(s_from)
            s_to[a] = 1
            return 1.0, s_to.astype(np.uint8)


    def __str__(self) -> str:
        return f"""
Task:
    Action Space: {self.num_actions}
    Features: \n{self.features}
"""

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    task = RIRLTask(features=np.random.random((5, 3)))
    print(task.actions, task.states)