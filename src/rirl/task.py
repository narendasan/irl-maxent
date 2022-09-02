from copy import deepcopy
from turtle import st
import numpy as np

class RIRLTask:
    def __init__(self, features):
        self.num_actions, self.num_features = np.shape(features)
        self.actions = self._generate_action_space()
        self.features = np.array(features)
        self.states = self._generate_state_space()
        assert(np.equal(self.states[len(self.states) - 1], np.ones((self.num_actions))).all())
        self.terminal_states = [self.states[len(self.states) - 1]]

    def _generate_state_space(self):
        max_val = (2 ** self.num_actions)
        state_list = [np.array(list(map(int, np.binary_repr(i, width=self.num_actions)))) for i in range(max_val)]
        states = np.vstack(state_list)
        assert states.shape == (max_val, self.num_actions)

        return states.astype(np.uint8)

    def _generate_action_space(self):
        return np.array(range(self.num_actions))

    @staticmethod
    def transition(s_from, a):
        # Action has been performed already
        if s_from[a] == 1:
            return 0.0, None

        else:
            s_to = deepcopy(s_from)
            s_to[a] = 1
            return 1.0, s_to.astype(np.uint8)



if __name__ == "__main__":
    task = RIRLTask(features=np.random.random((5, 3)))
    print(task.actions, task.states)